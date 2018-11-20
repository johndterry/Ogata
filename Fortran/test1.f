      program test1
          use adog_mod
          use input_func
          implicit none
          real(kind=8) z,bn,bx,eps,Q,ccut,qT,h,n,res,zv,error,zz
          common zv,qT,zz
          integer nu
          
          nu=0
          eps=0.01d0
          Q=1.5d0
          ccut=2.0d0
          qT=0.01d0
          zz=0.99d0
          
          call get_b_vals(lag_gauss,Q,eps,ccut,bn,bx)
          print *, bn,bx
          call get_ogata_params(bx,bn,qT/zz,h,n)
          print *, qT/zz,h,n
          call adog(lag_gauss,qT,Q,eps,ccut,nu,res)
          print *, 'ogata result is ', res
          print *, 'number of function calls is ', int(n)
          print *, 'analytic result is', lag_ana(qT,zz)
          error=abs((lag_ana(qT,zz)-res)/lag_ana(qT,zz))
          print *, 'error is ', error
          if (error.lt.0.01d0) then
            print *, ':)'
          else
            print *, ':('
          end if
          
      end program test1
