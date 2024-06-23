#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "../gemm.h"

double HPL_timer_walltime()
{
   struct timeval             tp;
   static long                start=0, startu;

   if( !start )
   {
      (void) gettimeofday( &tp, 0 );
      start  = tp.tv_sec;
      startu = tp.tv_usec;
      return( 0 );
   }
   (void) gettimeofday( &tp, 0 );

   return( (double)( tp.tv_sec - start ) +
           ( (double)( tp.tv_usec-startu ) / 1000000.0 ) );
}
