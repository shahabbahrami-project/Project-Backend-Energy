from django.urls import path

from user import views


app_name = 'user'

urlpatterns = [
    path('createuser/', views.CreateUserView.as_view(), name='createuser'),
    path('createsuperuser/', views.CreateSuperUserView.as_view(), name='createsuperuser'),
    path('token/', views.CreateTokenView.as_view(), name='token'),
    path('meuser/', views.ManageUserView.as_view(), name='meuser'),
    path('mesuperuser/', views.ManageSuperUserView.as_view(), name='mesuperuser'),
]
