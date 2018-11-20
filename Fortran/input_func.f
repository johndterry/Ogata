      module input_func
      
      contains
      
      real(kind=8) function lag_gauss(b)
      implicit none
      real(kind=8), intent(in) :: b
      real(kind=8) lambda,qT,zv,pi
      common zv,qT
      
      pi = 3.14159265359d0
      lambda = 1.5d0
      lag_gauss=b*dexp(-(b**2d0)*lambda**2d0/2d0)
      
      end function lag_gauss
      
      real(kind=8) function lag_gauss_q(b)
      implicit none
      real(kind=8), intent(in) :: b
      real(kind=8) lambda,qT,zv
      common zv,qT
      
      lambda = 1.5d0
      lag_gauss_q=b/qT**2*dexp(-(b**2d0)*(lambda**2d0)/2d0/qT**2)
      
      end function lag_gauss_q
      real(kind=8) function lag_ana(q,zz)
      implicit none
      real(kind=8), intent(in) :: q,zz
      real(kind=8) pi,lambda
      
      lambda = 1.5d0
      pi = 3.14159265359d0
      
      lag_ana=dexp(-q**2d0/zz**2d0/2d0/lambda**2)/2d0/pi/lambda**2d0
      
      end function lag_ana
      
      end module input_func
      
