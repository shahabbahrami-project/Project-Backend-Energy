from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import viewsets, mixins, status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated

from core.models import Sensor, Site, SensorType, SensorData, Device, DeviceType, DeviceData

from sites import serializers



# import pandas as pd
# import influxdb
# # to write pandas DataFrames into influx, or to read data into a pandas DataFrame
# from influxdb import DataFrameClient
# from datetime import timezone, datetime
# import pytz
# import certifi





from django.http import HttpResponse
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime, timedelta, time
import numpy as np
from numpy.random import randint
from sites.extra.PriceFunc import PriceFunc
from numpy import loadtxt
from datetime import timezone, datetime
from sites.MA_Algorithm.MAAlgorithm import moving_average
from sites.MPC.MPC import MPC
from sites.DQN.TestDRLGYM import ForwardDRLGYM
from sites.DQN.TrainDRLGYM import TrainDRLGYM
class BaseSiteAttrViewSet(viewsets.GenericViewSet,
                            mixins.ListModelMixin,
                            mixins.CreateModelMixin,
                            mixins.RetrieveModelMixin,
                            mixins.UpdateModelMixin,
                            mixins.DestroyModelMixin):
    """Base viewset for user owned site attributes"""
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)

    def get_queryset(self):
        """Return objects for the current authenticated user only"""
        assigned_only = bool(
            int(self.request.query_params.get('assigned_only', 0))
        )
        queryset = self.queryset
        if assigned_only:
            queryset = queryset.filter(site__isnull=False)

        return queryset.filter(
            user=self.request.user
        ).order_by('-name').distinct()

    def perform_create(self, serializer):
        """Create new object"""
        serializer.save(user=self.request.user)


class DeviceTypeViewSet(BaseSiteAttrViewSet):
    """Manage device type in the database"""
    queryset = DeviceType.objects.all()
    serializer_class = serializers.DeviceTypeSerializer


class DeviceViewSet(BaseSiteAttrViewSet):
    """Manage device in the database"""
    queryset = Device.objects.all()
    serializer_class = serializers.DeviceSerializer


class SensorTypeViewSet(BaseSiteAttrViewSet):
    """Manage sensor type in the database"""
    queryset = SensorType.objects.all()
    serializer_class = serializers.SensorTypeSerializer


class SensorViewSet(BaseSiteAttrViewSet):
    """Manage sensor in the database"""
    queryset = Sensor.objects.all()
    serializer_class = serializers.SensorSerializer


class SensorDataViewSet(BaseSiteAttrViewSet):
    """Manage sensor in the database"""
    queryset = SensorData.objects.all()
    serializer_class = serializers.SensorSerializer


class DeviceDataViewSet(BaseSiteAttrViewSet):
    """Manage sensor in the database"""
    queryset = DeviceData.objects.all()
    serializer_class = serializers.DeviceDataSerializer


class SiteViewSet(viewsets.ModelViewSet):
    """Manage sites in the database"""
    serializer_class = serializers.SiteSerializer
    queryset = Site.objects.all()
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)

    def _params_to_ints(self, qs):
        """Convert a list of string IDs to a list of integers"""
        return [int(str_id) for str_id in qs.split(',')]

    def get_queryset(self):
        """Return objects for the current authenticated user only"""
        sensors = self.request.query_params.get('sensors')
        queryset = self.queryset
        if sensors:
            sensor_ids = self._params_to_ints(sensors)
            queryset = queryset.filter(sensors__id__in=sensor_ids)

        return queryset.filter(user=self.request.user)

    def get_serializer_class(self):
        """Return appropriate serializer class for our request"""
        if self.action == 'retrieve':
            return serializers.SiteDetailSerializer
        elif self.action == 'upload_image':
            return serializers.SiteImageSerializer

        return self.serializer_class

    def perform_create(self, serializer):
        """Create a new site"""
        serializer.save(user=self.request.user)

    @action(methods=['POST'], detail=True, url_path='upload-image')
    def upload_image(self, request, pk=None):
        """Upload an image to a site"""
        site = self.get_object()
        serializer = self.get_serializer(
            site,
            data=request.data
        )

        if serializer.is_valid():
            serializer.save()
            return Response(
                serializer.data,
                status=status.HTTP_200_OK
            )

        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )

def periodselect(i):
    switcher={
            '1 Day': 1,
            '1 Week': 7,
            '1 Month': 30,
            '3 Months': 90,
            '6 Months': 120,
            '1 Year': 365,
         }
    return switcher.get(i,"Invalid period")

def myconverter(o):
    if isinstance(o, datetime):
        return o.isoformat()

@csrf_exempt
def SensorDataGeneration(request):
    z=np.exp(-300/130)
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    type = body["type"]
    period = body["period"]
    Appnum=1
    PowerSlice=0.1
    NumSlice=int(1+(1/PowerSlice))
    bet=0.8
    Day=10
    D=96
    NumDays=periodselect(period)
    T=D*NumDays
    HomeNum=1
    Price=np.zeros(T)
    Price=np.tile(PriceFunc(D),int(T/D))
    Flag = np.zeros(Appnum)

    Tmin = np.zeros((Appnum,T))
    Tmax = np.zeros((Appnum,T))

    Tairmin = 10*np.ones((Appnum,T))
    Tairmax = 30*np.ones((Appnum,T))

    Tsetmin = np.zeros((Appnum,T))
    Tsetmax = 30*np.ones((Appnum,T))

    w = 1*np.ones((Appnum,T))  #Unit:  cents/C
    test=1

    Tin_DRL = loadtxt('sites/extra/Tin.csv', delimiter=',')

    Cost_DRL= loadtxt('sites/extra/Cost.csv', delimiter=',')


    N=np.zeros((Appnum,T))
    for i in range(0,Appnum):
        for t in range(0, T-1):
            if t%96<=48:
                if t%3!=0:
                    N[i,t]=N[i,t-1]
                else:
                    N[i,t]= randint(1, 4)
                    while t>=0 and np.absolute(N[i,t]-N[i,t-1])>1:
                        N[i,t]= randint(1, 4)
            else:
                N[i,t]=0


    Tout = np.zeros((Appnum,T))
    for i in range(0,Appnum):
        for t in range(0, T-1):
            if t%96<=48:
                if t%3!=0:
                    Tout[i,t]=Tout[i,t-1]
                else:
                    Tout[i,t]=  0.25*randint(56, 64)
                    # while t>=0 and np.absolute(Tout[i,t]-Tout[i,t-1])>10:
                    #     Tout[i,t]= 0.25*randint(56, 64)
            else:
                if t%3!=0:
                    Tout[i,t]=Tout[i,t-1]
                else:
                    Tout[i,t]=  0.25*randint(44, 52)

    Tdes = np.zeros((Appnum,T))
    for i in range(0,Appnum):
        for t in range(0, T-1):
            if t%96<=48:
                if t%3!=0:
                    Tdes[i,t]=Tdes[i,t-1]
                else:
                    Tdes[i,t]= randint(24, 25)
            else:
                if t%3!=0:
                    Tdes[i,t]=Tdes[i,t-1]
                else:
                    Tdes[i,t]=  0

    Tset_manual = np.zeros((Appnum,T))
    for i in range(0,Appnum):
        for t in range(0, T-1):
            if t%96<=48:
                    Tset_manual[i,t]= 24
            else:
                    Tset_manual[i,t]= 5

    Tin_manual=np.zeros((Appnum,T))
    Tin_manual[:,0]=12
    for i in range(0,Appnum):
        for t in range(0, T-1):
            if Tin_manual[i,t]<Tset_manual[i,t]:
                Tin_manual[i,t+1]= Tin_manual[i,t]+(Tout[i,t]-Tin_manual[i,t])*z+(30-Tin_manual[i,t])*z
            else:
                Tin_manual[i,t+1]= Tin_manual[i,t]+(Tout[i,t]-Tin_manual[i,t])*z

    CumCost_manual=np.zeros((Appnum,T))
    CumCost_DRL=np.zeros((Appnum,T))
    Cost_manual=np.zeros((Appnum,T))
    for i in range(0,Appnum):
        for t in range(0, T-1):
            if Tin_manual[i,t]<Tset_manual[i,t]:

                Cost_manual[i,t]=Price[t]*np.absolute(30-Tin_manual[i,t])+N[i,t]*w[i,t]*np.absolute(Tdes[i,t]-Tin_manual[i,t])
            else:
                Cost_manual[i,t]=0*Price[t]*np.absolute(30-Tin_manual[i,t])+N[i,t]*w[i,t]*np.absolute(Tdes[i,t]-Tin_manual[i,t])
            CumCost_manual[i,t]= np.sum(Cost_manual[i,:])

    CumCost_DRL= np.cumsum(Cost_DRL)

    if type=="History":
        Tout_MA = np.zeros((Appnum,T))
        N_MA = np.zeros((Appnum,T))

    if type=="MA":
        Tout_MA = np.zeros((Appnum,T))
        Tout_MA[0,:]=moving_average(Tout[0,:], 12)
        N_MA = np.zeros((Appnum,T))
        N_MA[0,:]=moving_average(N[0,:], 4)

    # date_N_days_ago = datetime(2021, 3, 15, 0, 0, 0, 0)
    # date_now = datetime(2021, 3, 16, 0, 0, 0, 0)
    # date_N_days_ago = datetime.now() - timedelta(minutes=10*T)
    # date_now = datetime(2021, 3, 16, 0, 0, 0, 0)
    today = datetime.today()
    date_now=datetime.combine(today, datetime.min.time())
    date_N_days_ago = date_now - timedelta(minutes=15*T)
    # timearray = np.arange(date_N_days_ago, datetime.now(), timedelta(minutes=10)).astype(datetime)
    timearray = np.arange(date_N_days_ago, date_now, timedelta(minutes=15)).astype(datetime)
    meas=[]
    for t in range(T):
            row={'time':timearray[t],
                'outdoorTemp':Tout[0,t],
                'outdoorTempMA':Tout_MA[0,t],
                'desirableTemp':Tdes[0,t],
                'setpointManual':Tset_manual[0,t],
                'indoorTempManual':Tin_manual[0,t],
                'costManual':Cost_manual[0,t],
                'occupancy': N[0,t],
                'occupancyMA': N_MA[0,t],
                # 'indoorTempDRL':Tin_DRL[t],
                # 'costDRL':0.3*Cost_DRL[t],
                'price':Price[t],
                # 'cumcostDRL':0.3*CumCost_DRL[t],
                'cumcostManual':CumCost_manual[0,t]},
            meas.append(row)
    # print(meas)
    # N1=N.tolist()
    # data = json.dumps(N1)
    data = data=json.dumps(meas, default=myconverter)
    return HttpResponse(data)




@csrf_exempt
def SensorOnline(request):
    z=np.exp(-300/130)
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    FromHour = body["fromTimeSlot"]
    ToHour = body["toTimeSlot"]
    type= body["type"]
    weight= 10-body["weight"]
    Desire=body["desire"]
    Appnum=1
    PowerSlice=0.1
    NumSlice=int(1+(1/PowerSlice))
    bet=0.8
    Day=10
    D=96
    NumDays=1
    T=D*NumDays
    HomeNum=1
    Price=np.zeros(T)
    Price=np.tile(PriceFunc(D),int(T/D))
    Flag = np.zeros(Appnum)

    Tmin = np.zeros((Appnum,T))
    Tmax = np.zeros((Appnum,T))

    Tairmin = 10*np.ones((Appnum,T))
    Tairmax = 30*np.ones((Appnum,T))

    Tsetmin = np.zeros((Appnum,T))
    Tsetmax = 30*np.ones((Appnum,T))

    w = weight*np.ones((Appnum,T))  #Unit:  cents/C
    test=1

    # Tin_DRL = loadtxt('sites/extra/Tin.csv', delimiter=',')
    #
    # Cost_DRL= loadtxt('sites/extra/Cost.csv', delimiter=',')


    N=np.zeros((Appnum,T))
    for i in range(0,Appnum):
        for t in range(0, T-1):
            # if t%96<=48:
            #     if t%3!=0:
            #         N[i,t]=N[i,t-1]
            #     else:
            #         N[i,t]= randint(0, 4)
            #         while t>=0 and np.absolute(N[i,t]-N[i,t-1])>1:
            #             N[i,t]= randint(0, 4)
            # else:
            #     N[i,t]=0
            if t%96<FromHour:
                if t%3!=0:
                    N[i,t]=N[i,t-1]
                else:
                    N[i,t]= 0
            elif t%96>=FromHour and t%96<=ToHour:
                if t%3!=0:
                    N[i,t]=N[i,t-1]
                else:
                    N[i,t]=  randint(1, 4)
            else:
                N[i,t]=  0


    Tout = np.zeros((Appnum,T))
    for i in range(0,Appnum):
        for t in range(0, T-1):
            if t%96<=48:
                if t%3!=0:
                    Tout[i,t]=Tout[i,t-1]
                else:
                    Tout[i,t]=  0.25*randint(56, 64)
                    # while t>=0 and np.absolute(Tout[i,t]-Tout[i,t-1])>10:
                    #     Tout[i,t]= 0.25*randint(56, 64)
            else:
                if t%3!=0:
                    Tout[i,t]=Tout[i,t-1]
                else:
                    Tout[i,t]=  0.25*randint(44, 52)

    Tdes = np.zeros((Appnum,T))
    for i in range(0,Appnum):
        for t in range(0, T-1):
            if t%96<FromHour:
                if t%3!=0:
                    Tdes[i,t]=Tdes[i,t-1]
                else:
                    Tdes[i,t]= 0
            elif t%96>=FromHour and t%96<=ToHour:
                if t%3!=0:
                    Tdes[i,t]=Tdes[i,t-1]
                else:
                    Tdes[i,t]=  Desire
            else:
                Tdes[i,t]=  0
    Tset_manual = np.zeros((Appnum,T))
    for i in range(0,Appnum):
        for t in range(0, T-1):
            if t%96<FromHour:
                    Tset_manual[i,t]= 18
            elif t%96>=FromHour and t%96<=ToHour:
                    Tset_manual[i,t]=  Desire
            elif t%96>ToHour and t%96<=48:
                    Tset_manual[i,t]= 18
            elif t%96>ToHour and t%96>48:
                    Tset_manual[i,t]= 5

    Tin_manual=np.zeros((Appnum,T))
    Tin_manual[:,0]=12
    for i in range(0,Appnum):
        for t in range(0, T-1):
            if Tin_manual[i,t]<Tset_manual[i,t]:
                Tin_manual[i,t+1]= Tin_manual[i,t]+(Tout[i,t]-Tin_manual[i,t])*z+(30-Tin_manual[i,t])*z
            else:
                Tin_manual[i,t+1]= Tin_manual[i,t]+(Tout[i,t]-Tin_manual[i,t])*z

    CumCost_manual=np.zeros((Appnum,T))

    Cost_manual=np.zeros((Appnum,T))
    for i in range(0,Appnum):
        for t in range(0, T-1):
            if Tin_manual[i,t]<Tset_manual[i,t]:
                Cost_manual[i,t]=Price[t]*np.absolute(30-Tin_manual[i,t])+N[i,t]*w[i,t]*np.absolute(Tdes[i,t]-Tin_manual[i,t])
            else:
                Cost_manual[i,t]=0*Price[t]*np.absolute(30-Tin_manual[i,t])+N[i,t]*w[i,t]*np.absolute(Tdes[i,t]-Tin_manual[i,t])
            CumCost_manual[i,t]= np.sum(Cost_manual[i,:])

    Tout_MA = np.zeros((Appnum,T))
    N_MA = np.zeros((Appnum,T))
    Cost_MPC=np.zeros((Appnum,T))
    CumCost_MPC=np.zeros((Appnum,T))
    Tair_MPC = np.zeros((Appnum,T))
    Tin_MPC = np.zeros((Appnum,T))
    Tset_MPC = np.zeros((Appnum,T))
    if type=="MPC" or type=="DRLMPCManual":

        for t in range(T):
            if t>=1:
                Tair_MPC[0,t], Tin_MPC[0,t], Tset_MPC[0,t], Cost_MPC[0,t]=MPC(Tin_MPC[0,t-1],Tout[0,t],Tdes[0,t],N[0,t],Price[t],w[0,t])
            else:
                Tair_MPC[0,t], Tin_MPC[0,t], Tset_MPC[0,t], Cost_MPC[0,t]=MPC(Tin_manual[0,t],Tout[0,t],Tdes[0,t],N[0,t],Price[t],w[0,t])
            if np.absolute(Tair_MPC[0,t]-Tin_MPC[0,t])<=0.5:
                Tset_MPC[0,t]=Tin_MPC[0,t]/2
            else:
                Tset_MPC[0,t]=np.minimum(Tin_MPC[0,t]+3, 30)
            CumCost_MPC[0,t]= np.sum(Cost_MPC[0,:])

    Cost_DRL=np.zeros((Appnum,T))
    CumCost_DRL=np.zeros((Appnum,T))
    Tair_DRL = np.zeros((Appnum,T))
    Tin_DRL = np.zeros((Appnum,T))
    Tset_DRL = np.zeros((Appnum,T))
    print(type)
    if type=="DRL" or type=="DRLMPCManual":
        print(FromHour)
        print(ToHour)
        print(weight)
        print(Desire)
        flag=TrainDRLGYM(FromHour,ToHour,weight,Desire)
        print(flag)
        for t in range(T):
            print('time=',t)
            if t>=1:
                sample=[Tin_DRL[0,t-1],Tdes[0,t],Tout[0,t],Price[t],N[0,t],t]
                Tair_DRL[0,t], Tin_DRL[0,t], Tset_DRL[0,t], Cost_DRL[0,t]=ForwardDRLGYM(FromHour,ToHour,weight,Desire, sample)
            else:
                sample=[12,Tdes[0,t],Tout[0,t],Price[t],N[0,t],t]
                Tair_DRL[0,t], Tin_DRL[0,t], Tset_DRL[0,t], Cost_DRL[0,t]=ForwardDRLGYM(FromHour,ToHour,weight,Desire, sample)
            CumCost_DRL[0,t]= np.sum(Cost_DRL[0,:])
    today = datetime.today()
    date_now=datetime.combine(today, datetime.min.time())
    date_N_days_ago = date_now - timedelta(minutes=15*T)
    # timearray = np.arange(date_N_days_ago, datetime.now(), timedelta(minutes=10)).astype(datetime)
    timearray = np.arange(date_N_days_ago, date_now, timedelta(minutes=15)).astype(datetime)
    meas=[]
    for t in range(T):
            row={'time':timearray[t],
                'outdoorTemp':Tout[0,t],
                'outdoorTempMA':Tout_MA[0,t],
                'desirableTemp':Tdes[0,t],
                'setpointManual':Tset_manual[0,t],
                'indoorTempManual':Tin_manual[0,t],
                'costManual':Cost_manual[0,t],
                'costMPC':Cost_MPC[0,t],
                'occupancy': N[0,t],
                'occupancyMA': N_MA[0,t],
                'setpointMPC':Tset_MPC[0,t],
                'indoorTempMPC':Tin_MPC[0,t],
                'airTempMPC':Tair_MPC[0,t],
                'indoorTempDRL':Tin_DRL[0,t],
                'costDRL':Cost_DRL[0,t],
                'setpointDRL':Tset_DRL[0,t],
                'airTempDRL':Tair_DRL[0,t],
                'price':Price[t],
                'cumcostDRL':CumCost_DRL[0,t],
                'cumcostManual':CumCost_manual[0,t],
                'cumcostMPC':CumCost_MPC[0,t]},
            meas.append(row)
    data = data=json.dumps(meas, default=myconverter)
    return HttpResponse(data)
