#include <stdio.h>
#include <cutil.h>
#include <cutil_inline.h>
#include "eggMotionEstimation.h"


//---------------------------------------------------------------------------------------------------
short*  pucOrgStart;                                        
short*  pucRefStart;
int     refImageSize;        
int     orgImageSize;

short*  d_yRef;
short*  d_yOrg;




texture<short , 2, cudaReadModeElementType> tex_ref;
cudaArray* cu_array_ref;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<short>();
dim3 grid4x4Mv((IMAGE_WIDTH/4), IMAGE_HEIGHT/4);
dim3 grid8x4Mv((IMAGE_WIDTH/8), IMAGE_HEIGHT/4);
dim3 grid4x8Mv((IMAGE_WIDTH/4), IMAGE_HEIGHT/8);
dim3 grid8x8Mv((IMAGE_WIDTH/8), IMAGE_HEIGHT/8);
dim3 grid8x16Mv((IMAGE_WIDTH/8), IMAGE_HEIGHT/16);
dim3 grid16x8Mv((IMAGE_WIDTH/16), IMAGE_HEIGHT/8);
dim3 grid16x16Mv((IMAGE_WIDTH/16), IMAGE_HEIGHT/16);

dim3 gridInterpHpel((IMAGE_WIDTH + SEARCH_RANGE*2)/INTERPOLATEDIM, (IMAGE_HEIGHT + SEARCH_RANGE*2));
dim3 gridInterpQpel(((IMAGE_WIDTH + SEARCH_RANGE*2))/INTERPOLATEDIM, (IMAGE_HEIGHT + SEARCH_RANGE*2));


//---------------------------------------------------------------------------------------------------
__global__ void kernelPmods4x4Mv( int orgStride, short* yOrg,
                                  short* d_mvX, short* d_mvY, uint* d_sad){
        uint tid = threadIdx.x;
        uint bx  = blockIdx.x;
        uint by  = blockIdx.y;
        
        
        short pucCurX = SEARCH_RANGE;
        short pucCurY = SEARCH_RANGE;
        short* pucOrg;


        __shared__ short2 s_mv[BLOCK_DIM];
        __shared__ uint   s_sad[BLOCK_DIM];
        __shared__ short  s_orgMb[16];

        short mvY;
        uint  curSad;

        pucOrg = yOrg + Shift2Next4x4Block(orgStride, by, bx);
        pucCurX += bx*4;
        pucCurY += by*4;
        
        pucCurX += (tid - SEARCH_RANGE);
        
        
        short  pucCurY1;
        short  pucCurY2;
        short  pucCurY3;
        short  pucCurY4;
        short  pucCurY5;
        short  pucCurY6;
        
        uint    cur1Sad = 0;
        uint    cur2Sad = 0;
        uint    cur3Sad = 0;
        uint    cur4Sad = 0;
        uint    cur5Sad = 0;
        uint    cur6Sad = 0;
        

        mvY = 0;
        curSad = 0;
        
       
        pucCurY1  = pucCurY + P1;
        pucCurY2  = pucCurY + P2;
        pucCurY3  = pucCurY + P3;
        pucCurY4  = pucCurY + P4;
        pucCurY5  = pucCurY + P5;
        pucCurY6  = pucCurY + P6;

        if(tid < 16){
                s_orgMb[tid] = pucOrg[Index2DAddress(orgStride, tid>>2, tid&3)];
        }
        __syncthreads();
        
        for(ushort m = 0; m < 4; m++){
                for(ushort n = 0; n < 4; n++){
                        curSad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY + m), s_orgMb[Index2DAddress(4, m, n)], curSad);
                        cur1Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(4, m, n)], cur1Sad);
                        cur2Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(4, m, n)], cur2Sad);
                        cur3Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY3 + m), s_orgMb[Index2DAddress(4, m, n)], cur3Sad);
                        cur4Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY4 + m), s_orgMb[Index2DAddress(4, m, n)], cur4Sad);
                        cur5Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY5 + m), s_orgMb[Index2DAddress(4, m, n)], cur5Sad);
                        cur6Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY6 + m), s_orgMb[Index2DAddress(4, m, n)], cur6Sad);
                }
        }

        
        short distance = 0;     // distance is 0 for the central position 
        if(curSad > cur1Sad) { distance = P1; curSad = cur1Sad;}
        if(curSad > cur2Sad) { distance = P2; curSad = cur2Sad;}
        if(curSad > cur3Sad) { distance = P3; curSad = cur3Sad;}
        if(curSad > cur4Sad) { distance = P4; curSad = cur4Sad;}
        if(curSad > cur5Sad) { distance = P5; curSad = cur5Sad;}
        if(curSad > cur6Sad) { distance = P6; curSad = cur6Sad;}
        
        mvY += distance;
        
        for(ushort s = STEP_SIZE; s > 0; s >>= 1){

                pucCurY = pucCurY + distance;
                cur1Sad = 0;
                cur2Sad = 0;
        
                pucCurY1  = pucCurY + (-s);
                pucCurY2  = pucCurY + s;
                for(ushort m = 0; m < 4; m++){
                        for(ushort n = 0; n < 4; n++){
                                
                                cur1Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(4, m, n)], cur1Sad);
                                cur2Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(4, m, n)], cur2Sad);
                        }
                }

                distance = 0;     // distance is 0 for the central position
                if(curSad > cur1Sad) { distance = -s; curSad = cur1Sad;}
                if(curSad > cur2Sad) { distance = s; curSad = cur2Sad;}
                mvY += distance;          

        }
        
        s_mv[tid].y = mvY;
        s_sad[tid] = curSad;

        s_mv[tid].x = tid - SEARCH_RANGE;
                                


#if SEARCH_RANGE > 32
        __syncthreads();

        if(tid < 64){
                if(s_sad[tid] > s_sad[tid + 64]){
                        s_sad[tid] = s_sad[tid + 64];
                        s_mv[tid].y = s_mv[tid + 64].y;
                        s_mv[tid].x = s_mv[tid + 64].x;
                }
        }
#endif
        __syncthreads();
        
        if(tid < 32){   //in 1 warp, needless __syncthreads()  
                if(s_sad[tid] > s_sad[tid + 32]){
                        s_sad[tid] = s_sad[tid + 32];
                        s_mv[tid].y = s_mv[tid + 32].y;
                        s_mv[tid].x = s_mv[tid + 32].x;
                }
                if(s_sad[tid] > s_sad[tid + 16]){
                        s_sad[tid] = s_sad[tid + 16];
                        s_mv[tid].y = s_mv[tid + 16].y;
                        s_mv[tid].x = s_mv[tid + 16].x;
                }
                if(s_sad[tid] > s_sad[tid + 8]){
                        s_sad[tid] = s_sad[tid + 8];
                        s_mv[tid].y = s_mv[tid + 8].y;
                        s_mv[tid].x = s_mv[tid + 8].x;
                }
                if(s_sad[tid] > s_sad[tid + 4]){
                        s_sad[tid] = s_sad[tid + 4];
                        s_mv[tid].y = s_mv[tid + 4].y;
                        s_mv[tid].x = s_mv[tid + 4].x;
                }
                if(s_sad[tid] > s_sad[tid + 2]){
                        s_sad[tid] = s_sad[tid + 2];
                        s_mv[tid].y = s_mv[tid + 2].y;
                        s_mv[tid].x = s_mv[tid + 2].x;
                }
                if(s_sad[0] > s_sad[1]){
                        d_sad[by*(IMAGE_WIDTH/4) + bx] = s_sad[1];
                        d_mvY[by*(IMAGE_WIDTH/4) + bx] = s_mv[1].y;
                        d_mvX[by*(IMAGE_WIDTH/4) + bx] = s_mv[1].x;
                }else{
                        d_sad[by*(IMAGE_WIDTH/4) + bx] = s_sad[0];
                        d_mvY[by*(IMAGE_WIDTH/4) + bx] = s_mv[0].y;
                        d_mvX[by*(IMAGE_WIDTH/4) + bx] = s_mv[0].x;
                }
        }
}
//---------------------------------------------------------------------------------------------------
__global__ void kernelPmods4x8Mv( int orgStride, short* yOrg,
                                  short* d_mvX, short* d_mvY, uint* d_sad){


        uint tid = threadIdx.x;
        uint bx  = blockIdx.x;
        uint by  = blockIdx.y;
        
      
        short pucCurX = SEARCH_RANGE;
        short pucCurY = SEARCH_RANGE;
        short* pucOrg;
        
         
        __shared__ short2 s_mv[BLOCK_DIM];
        __shared__ uint   s_sad[BLOCK_DIM];
        __shared__ short  s_orgMb[32];
        short mvY;
        uint  curSad;
                        
                        

                           
        pucOrg = yOrg + Shift2Next4x8Block(orgStride, by, bx);
        pucCurX += bx*4;
        pucCurY += by*8;
        
        pucCurX += (tid - SEARCH_RANGE);
        
        short  pucCurY1;
        short  pucCurY2;
        short  pucCurY3;
        short  pucCurY4;
        short  pucCurY5;
        short  pucCurY6;
        
        uint    cur1Sad = 0;
        uint    cur2Sad = 0;
        uint    cur3Sad = 0;
        uint    cur4Sad = 0;
        uint    cur5Sad = 0;
        uint    cur6Sad = 0;
        

        mvY = 0;
        curSad = 0;
        
        
        pucCurY1  = pucCurY + P1;
        pucCurY2  = pucCurY + P2;
        pucCurY3  = pucCurY + P3;
        pucCurY4  = pucCurY + P4;
        pucCurY5  = pucCurY + P5;
        pucCurY6  = pucCurY + P6;

        if(tid < 32){
                s_orgMb[tid] = pucOrg[Index2DAddress(orgStride, tid>>2, tid&3)];
        }
        __syncthreads(); 
        
        for(ushort m = 0; m < 8; m++){
                for(ushort n = 0; n < 4; n++){
                        
                        curSad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY + m), s_orgMb[Index2DAddress(4, m, n)], curSad);
                        cur1Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(4, m, n)], cur1Sad);
                        cur2Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(4, m, n)], cur2Sad);
                        cur3Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY3 + m), s_orgMb[Index2DAddress(4, m, n)], cur3Sad);
                        cur4Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY4 + m), s_orgMb[Index2DAddress(4, m, n)], cur4Sad);
                        cur5Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY5 + m), s_orgMb[Index2DAddress(4, m, n)], cur5Sad);
                        cur6Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY6 + m), s_orgMb[Index2DAddress(4, m, n)], cur6Sad);
                }
        }

        
        short distance = 0;    
        if(curSad > cur1Sad) { distance = P1; curSad = cur1Sad;}
        if(curSad > cur2Sad) { distance = P2; curSad = cur2Sad;}
        if(curSad > cur3Sad) { distance = P3; curSad = cur3Sad;}
        if(curSad > cur4Sad) { distance = P4; curSad = cur4Sad;}
        if(curSad > cur5Sad) { distance = P5; curSad = cur5Sad;}
        if(curSad > cur6Sad) { distance = P6; curSad = cur6Sad;}
        
        mvY += distance;
        
        for(ushort s = STEP_SIZE; s > 0; s >>= 1){

                
                pucCurY = pucCurY + distance;
                cur1Sad = 0;
                cur2Sad = 0;
        
                pucCurY1  = pucCurY + (-s);
                pucCurY2  = pucCurY + s;
                for(ushort m = 0; m < 8; m++){
                        for(ushort n = 0; n < 4; n++){
                                
                                cur1Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(4, m, n)], cur1Sad);
                                cur2Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(4, m, n)], cur2Sad);
                        }
                }

                distance = 0;  
                if(curSad > cur1Sad) { distance = -s; curSad = cur1Sad;}
                if(curSad > cur2Sad) { distance = s; curSad = cur2Sad;}
                mvY += distance;          

        }
        
        s_mv[tid].y = mvY;
        s_sad[tid] = curSad;

        s_mv[tid].x = tid - SEARCH_RANGE;
                                
                       


#if SEARCH_RANGE > 32
        __syncthreads();

        if(tid < 64){
                if(s_sad[tid] > s_sad[tid + 64]){
                        s_sad[tid] = s_sad[tid + 64];
                        s_mv[tid].y = s_mv[tid + 64].y;
                        s_mv[tid].x = s_mv[tid + 64].x;
                }
        }
#endif
        __syncthreads();
        
        if(tid < 32){   //in 1 warp, needless __syncthreads()  
                if(s_sad[tid] > s_sad[tid + 32]){
                        s_sad[tid] = s_sad[tid + 32];
                        s_mv[tid].y = s_mv[tid + 32].y;
                        s_mv[tid].x = s_mv[tid + 32].x;
                }
                if(s_sad[tid] > s_sad[tid + 16]){
                        s_sad[tid] = s_sad[tid + 16];
                        s_mv[tid].y = s_mv[tid + 16].y;
                        s_mv[tid].x = s_mv[tid + 16].x;
                }
                if(s_sad[tid] > s_sad[tid + 8]){
                        s_sad[tid] = s_sad[tid + 8];
                        s_mv[tid].y = s_mv[tid + 8].y;
                        s_mv[tid].x = s_mv[tid + 8].x;
                }
                if(s_sad[tid] > s_sad[tid + 4]){
                        s_sad[tid] = s_sad[tid + 4];
                        s_mv[tid].y = s_mv[tid + 4].y;
                        s_mv[tid].x = s_mv[tid + 4].x;
                }
                if(s_sad[tid] > s_sad[tid + 2]){
                        s_sad[tid] = s_sad[tid + 2];
                        s_mv[tid].y = s_mv[tid + 2].y;
                        s_mv[tid].x = s_mv[tid + 2].x;
                }
                if(s_sad[0] > s_sad[1]){
                        d_sad[by*(IMAGE_WIDTH/4) + bx] = s_sad[1];
                        d_mvY[by*(IMAGE_WIDTH/4) + bx] = s_mv[1].y;
                        d_mvX[by*(IMAGE_WIDTH/4) + bx] = s_mv[1].x;
                }else{
                        d_sad[by*(IMAGE_WIDTH/4) + bx] = s_sad[0];
                        d_mvY[by*(IMAGE_WIDTH/4) + bx] = s_mv[0].y;
                        d_mvX[by*(IMAGE_WIDTH/4) + bx] = s_mv[0].x;
                }
        }
        
}
//---------------------------------------------------------------------------------------------------
__global__ void kernelPmods8x4Mv( int orgStride, short* yOrg,
                                  short* d_mvX, short* d_mvY, uint* d_sad){


        uint tid = threadIdx.x;
        uint bx  = blockIdx.x;
        uint by  = blockIdx.y;
        
        short pucCurX = SEARCH_RANGE;
        short pucCurY = SEARCH_RANGE;
        short* pucOrg;
        
                        
        __shared__ short2 s_mv[BLOCK_DIM];
        __shared__ uint   s_sad[BLOCK_DIM];
        __shared__ short  s_orgMb[32];
        short mvY;
        uint  curSad;
                     
        pucOrg = yOrg + Shift2Next8x4Block(orgStride, by, bx);
        pucCurX += bx*8;
        pucCurY += by*4;
        
        pucCurX += (tid - SEARCH_RANGE);
        
        short  pucCurY1;
        short  pucCurY2;
        short  pucCurY3;
        short  pucCurY4;
        short  pucCurY5;
        short  pucCurY6;
        
        uint    cur1Sad = 0;
        uint    cur2Sad = 0;
        uint    cur3Sad = 0;
        uint    cur4Sad = 0;
        uint    cur5Sad = 0;
        uint    cur6Sad = 0;
        

        mvY = 0;
        curSad = 0;

        pucCurY1  = pucCurY + P1;
        pucCurY2  = pucCurY + P2;
        pucCurY3  = pucCurY + P3;
        pucCurY4  = pucCurY + P4;
        pucCurY5  = pucCurY + P5;
        pucCurY6  = pucCurY + P6;

        if(tid < 32){
                s_orgMb[tid] = pucOrg[Index2DAddress(orgStride, tid>>3, tid&7)];
        }
        __syncthreads();
        
        for(int m = 0; m < 4; m++){
                for(int n = 0; n < 8; n++){
                        
                        curSad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY + m), s_orgMb[Index2DAddress(8, m, n)], curSad);
                        cur1Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(8, m, n)], cur1Sad);
                        cur2Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(8, m, n)], cur2Sad);
                        cur3Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY3 + m), s_orgMb[Index2DAddress(8, m, n)], cur3Sad);
                        cur4Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY4 + m), s_orgMb[Index2DAddress(8, m, n)], cur4Sad);
                        cur5Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY5 + m), s_orgMb[Index2DAddress(8, m, n)], cur5Sad);
                        cur6Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY6 + m), s_orgMb[Index2DAddress(8, m, n)], cur6Sad);
                        
                }
        }

        
        short distance = 0;   
        if(curSad > cur1Sad) { distance = P1; curSad = cur1Sad;}
        if(curSad > cur2Sad) { distance = P2; curSad = cur2Sad;}
        if(curSad > cur3Sad) { distance = P3; curSad = cur3Sad;}
        if(curSad > cur4Sad) { distance = P4; curSad = cur4Sad;}
        if(curSad > cur5Sad) { distance = P5; curSad = cur5Sad;}
        if(curSad > cur6Sad) { distance = P6; curSad = cur6Sad;}
        
        mvY += distance;
        
        for(int s = STEP_SIZE; s > 0; s >>= 1){

                
                pucCurY = pucCurY + distance;
                cur1Sad = 0;
                cur2Sad = 0;
        
                
                pucCurY1  = pucCurY + (-s);
                pucCurY2  = pucCurY + s;
                for(int m = 0; m < 4; m++){
                        for(int n = 0; n < 8; n++){
                                
                                cur1Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(8, m, n)], cur1Sad);
                                cur2Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(8, m, n)], cur2Sad);
                                
                        }
                }

                distance = 0;   
                if(curSad > cur1Sad) { distance = -s; curSad = cur1Sad;}
                if(curSad > cur2Sad) { distance = s; curSad = cur2Sad;}
                mvY += distance;          

        }
        
        s_mv[tid].y = mvY;
        s_sad[tid] = curSad;

        s_mv[tid].x = tid - SEARCH_RANGE;
                                
                        


#if SEARCH_RANGE > 32
        __syncthreads();

        if(tid < 64){
                if(s_sad[tid] > s_sad[tid + 64]){
                        s_sad[tid] = s_sad[tid + 64];
                        s_mv[tid].y = s_mv[tid + 64].y;
                        s_mv[tid].x = s_mv[tid + 64].x;
                }
        }
#endif
        __syncthreads();
        
        if(tid < 32){   //in 1 warp, needless __syncthreads() 
                if(s_sad[tid] > s_sad[tid + 32]){
                        s_sad[tid] = s_sad[tid + 32];
                        s_mv[tid].y = s_mv[tid + 32].y;
                        s_mv[tid].x = s_mv[tid + 32].x;
                }
                if(s_sad[tid] > s_sad[tid + 16]){
                        s_sad[tid] = s_sad[tid + 16];
                        s_mv[tid].y = s_mv[tid + 16].y;
                        s_mv[tid].x = s_mv[tid + 16].x;
                }
                if(s_sad[tid] > s_sad[tid + 8]){
                        s_sad[tid] = s_sad[tid + 8];
                        s_mv[tid].y = s_mv[tid + 8].y;
                        s_mv[tid].x = s_mv[tid + 8].x;
                }
                if(s_sad[tid] > s_sad[tid + 4]){
                        s_sad[tid] = s_sad[tid + 4];
                        s_mv[tid].y = s_mv[tid + 4].y;
                        s_mv[tid].x = s_mv[tid + 4].x;
                }
                if(s_sad[tid] > s_sad[tid + 2]){
                        s_sad[tid] = s_sad[tid + 2];
                        s_mv[tid].y = s_mv[tid + 2].y;
                        s_mv[tid].x = s_mv[tid + 2].x;
                }
                if(s_sad[0] > s_sad[1]){
                        d_sad[by*(IMAGE_WIDTH/8) + bx] = s_sad[1];
                        d_mvY[by*(IMAGE_WIDTH/8) + bx] = s_mv[1].y;
                        d_mvX[by*(IMAGE_WIDTH/8) + bx] = s_mv[1].x;
                }else{
                        d_sad[by*(IMAGE_WIDTH/8) + bx] = s_sad[0];
                        d_mvY[by*(IMAGE_WIDTH/8) + bx] = s_mv[0].y;
                        d_mvX[by*(IMAGE_WIDTH/8) + bx] = s_mv[0].x;
                }
        }
        
}
//---------------------------------------------------------------------------------------------------
__global__ void kernelPmods8x8Mv( int orgStride, short* yOrg,
                                  short* d_mvX, short* d_mvY, uint* d_sad){


        uint tid = threadIdx.x;
        uint bx  = blockIdx.x;
        uint by  = blockIdx.y;
        
        short pucCurX = SEARCH_RANGE;
        short pucCurY = SEARCH_RANGE;
        short* pucOrg;
        
                        
        __shared__ short2 s_mv[BLOCK_DIM];
        __shared__ uint   s_sad[BLOCK_DIM];
        __shared__ short  s_orgMb[64];
        short mvY;
        uint  curSad;
        
                       
        pucOrg = yOrg + Shift2Next8x8Block(orgStride, by, bx);
        pucCurX += bx*8;
        pucCurY += by*8;
        
        pucCurX += (tid - SEARCH_RANGE);
        
        
        short  pucCurY1;
        short  pucCurY2;
        short  pucCurY3;
        short  pucCurY4;
        short  pucCurY5;
        short  pucCurY6;
        
        uint    cur1Sad = 0;
        uint    cur2Sad = 0;
        uint    cur3Sad = 0;
        uint    cur4Sad = 0;
        uint    cur5Sad = 0;
        uint    cur6Sad = 0;
        

        mvY = 0;
        curSad = 0;
        
       
        pucCurY1  = pucCurY + P1;
        pucCurY2  = pucCurY + P2;
        pucCurY3  = pucCurY + P3;
        pucCurY4  = pucCurY + P4;
        pucCurY5  = pucCurY + P5;
        pucCurY6  = pucCurY + P6;

        if(tid < 64){
                s_orgMb[tid] = pucOrg[Index2DAddress(orgStride, tid>>3, tid&7)];
        }
        __syncthreads(); 
        
        for(int m = 0; m < 8; m++){
                for(int n = 0; n < 8; n++){
                        
                        curSad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY + m), s_orgMb[Index2DAddress(8, m, n)], curSad);
                        cur1Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(8, m, n)], cur1Sad);
                        cur2Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(8, m, n)], cur2Sad);
                        cur3Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY3 + m), s_orgMb[Index2DAddress(8, m, n)], cur3Sad);
                        cur4Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY4 + m), s_orgMb[Index2DAddress(8, m, n)], cur4Sad);
                        cur5Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY5 + m), s_orgMb[Index2DAddress(8, m, n)], cur5Sad);
                        cur6Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY6 + m), s_orgMb[Index2DAddress(8, m, n)], cur6Sad);
                }
        }

       
        short distance = 0;  
        if(curSad > cur1Sad) { distance = P1; curSad = cur1Sad;}
        if(curSad > cur2Sad) { distance = P2; curSad = cur2Sad;}
        if(curSad > cur3Sad) { distance = P3; curSad = cur3Sad;}
        if(curSad > cur4Sad) { distance = P4; curSad = cur4Sad;}
        if(curSad > cur5Sad) { distance = P5; curSad = cur5Sad;}
        if(curSad > cur6Sad) { distance = P6; curSad = cur6Sad;}
        
        mvY += distance;
        
        for(int s = STEP_SIZE; s > 0; s >>= 1){

         
                pucCurY = pucCurY + distance;
                cur1Sad = 0;
                cur2Sad = 0;
        
                pucCurY1  = pucCurY + (-s);
                pucCurY2  = pucCurY + s;
                for(int m = 0; m < 8; m++){
                        for(int n = 0; n < 8; n++){
                                
                                cur1Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(8, m, n)], cur1Sad);
                                cur2Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(8, m, n)], cur2Sad);
                        }
                }

                distance = 0;     
                if(curSad > cur1Sad) { distance = -s; curSad = cur1Sad;}
                if(curSad > cur2Sad) { distance = s; curSad = cur2Sad;}
                mvY += distance;          

        }
        
        s_mv[tid].y = mvY;
        s_sad[tid] = curSad;

        s_mv[tid].x = tid - SEARCH_RANGE;
                                
                        
#if SEARCH_RANGE > 32
        __syncthreads();

        if(tid < 64){
                if(s_sad[tid] > s_sad[tid + 64]){
                        s_sad[tid] = s_sad[tid + 64];
                        s_mv[tid].y = s_mv[tid + 64].y;
                        s_mv[tid].x = s_mv[tid + 64].x;
                }
        }
#endif
        __syncthreads();
        
        if(tid < 32){   //in 1 warp, needless __syncthreads()   
                if(s_sad[tid] > s_sad[tid + 32]){
                        s_sad[tid] = s_sad[tid + 32];
                        s_mv[tid].y = s_mv[tid + 32].y;
                        s_mv[tid].x = s_mv[tid + 32].x;
                }
                if(s_sad[tid] > s_sad[tid + 16]){
                        s_sad[tid] = s_sad[tid + 16];
                        s_mv[tid].y = s_mv[tid + 16].y;
                        s_mv[tid].x = s_mv[tid + 16].x;
                }
                if(s_sad[tid] > s_sad[tid + 8]){
                        s_sad[tid] = s_sad[tid + 8];
                        s_mv[tid].y = s_mv[tid + 8].y;
                        s_mv[tid].x = s_mv[tid + 8].x;
                }
                if(s_sad[tid] > s_sad[tid + 4]){
                        s_sad[tid] = s_sad[tid + 4];
                        s_mv[tid].y = s_mv[tid + 4].y;
                        s_mv[tid].x = s_mv[tid + 4].x;
                }
                if(s_sad[tid] > s_sad[tid + 2]){
                        s_sad[tid] = s_sad[tid + 2];
                        s_mv[tid].y = s_mv[tid + 2].y;
                        s_mv[tid].x = s_mv[tid + 2].x;
                }
                if(s_sad[0] > s_sad[1]){
                        d_sad[by*(IMAGE_WIDTH/8) + bx] = s_sad[1];
                        d_mvY[by*(IMAGE_WIDTH/8) + bx] = s_mv[1].y;
                        d_mvX[by*(IMAGE_WIDTH/8) + bx] = s_mv[1].x;
                }else{
                        d_sad[by*(IMAGE_WIDTH/8) + bx] = s_sad[0];
                        d_mvY[by*(IMAGE_WIDTH/8) + bx] = s_mv[0].y;
                        d_mvX[by*(IMAGE_WIDTH/8) + bx] = s_mv[0].x;
                }
        }
        
}
//---------------------------------------------------------------------------------------------------
__global__ void kernelPmods8x16Mv( int orgStride, short* yOrg,
                                   short* d_mvX, short* d_mvY, uint* d_sad){


        uint tid = threadIdx.x;
        uint bx  = blockIdx.x;
        uint by  = blockIdx.y;
        
       
        short pucCurX = SEARCH_RANGE;
        short pucCurY = SEARCH_RANGE;
        short* pucOrg;
        
        
        __shared__ short2 s_mv[BLOCK_DIM];
        __shared__ uint   s_sad[BLOCK_DIM];
        __shared__ short  s_orgMb[128];
        short mvY;
        uint  curSad;
        
        pucOrg = yOrg + Shift2Next8x16Block(orgStride, by, bx);
        pucCurX += bx*8;
        pucCurY += by*16;
       
        pucCurX += (tid - SEARCH_RANGE);
        
        
        short  pucCurY1;
        short  pucCurY2;
        short  pucCurY3;
        short  pucCurY4;
        short  pucCurY5;
        short  pucCurY6;
        
        uint    cur1Sad = 0;
        uint    cur2Sad = 0;
        uint    cur3Sad = 0;
        uint    cur4Sad = 0;
        uint    cur5Sad = 0;
        uint    cur6Sad = 0;
        

        mvY = 0;
        curSad = 0;
        
        pucCurY1  = pucCurY + P1;
        pucCurY2  = pucCurY + P2;
        pucCurY3  = pucCurY + P3;
        pucCurY4  = pucCurY + P4;
        pucCurY5  = pucCurY + P5;
        pucCurY6  = pucCurY + P6;

        if(tid < 128){
                s_orgMb[tid] = pucOrg[Index2DAddress(orgStride, tid>>3, tid&7)];
        }
        __syncthreads(); 
        
        for(int m = 0; m < 16; m++){
                for(int n = 0; n < 8; n++){

                        curSad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY + m), s_orgMb[Index2DAddress(8, m, n)], curSad);
                        cur1Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(8, m, n)], cur1Sad);
                        cur2Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(8, m, n)], cur2Sad);
                        cur3Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY3 + m), s_orgMb[Index2DAddress(8, m, n)], cur3Sad);
                        cur4Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY4 + m), s_orgMb[Index2DAddress(8, m, n)], cur4Sad);
                        cur5Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY5 + m), s_orgMb[Index2DAddress(8, m, n)], cur5Sad);
                        cur6Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY6 + m), s_orgMb[Index2DAddress(8, m, n)], cur6Sad);
                }
        }

        
        short distance = 0;     
        if(curSad > cur1Sad) { distance = P1; curSad = cur1Sad;}
        if(curSad > cur2Sad) { distance = P2; curSad = cur2Sad;}
        if(curSad > cur3Sad) { distance = P3; curSad = cur3Sad;}
        if(curSad > cur4Sad) { distance = P4; curSad = cur4Sad;}
        if(curSad > cur5Sad) { distance = P5; curSad = cur5Sad;}
        if(curSad > cur6Sad) { distance = P6; curSad = cur6Sad;}
        
        mvY += distance;
        
        for(int s = STEP_SIZE; s > 0; s >>= 1){

                
                pucCurY = pucCurY + distance;
                cur1Sad = 0;
                cur2Sad = 0;
        
                pucCurY1  = pucCurY + (-s);
                pucCurY2  = pucCurY + s;
                for(int m = 0; m < 16; m++){
                        for(int n = 0; n < 8; n++){
                                
                                cur1Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(8, m, n)], cur1Sad);
                                cur2Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(8, m, n)], cur2Sad);
                        }
                }

                distance = 0;    
                if(curSad > cur1Sad) { distance = -s; curSad = cur1Sad;}
                if(curSad > cur2Sad) { distance = s; curSad = cur2Sad;}
                mvY += distance;          

        }
        
        s_mv[tid].y = mvY;
        s_sad[tid] = curSad;

        s_mv[tid].x = tid - SEARCH_RANGE;
                                
                       
#if SEARCH_RANGE > 32
        __syncthreads();

        if(tid < 64){
                if(s_sad[tid] > s_sad[tid + 64]){
                        s_sad[tid] = s_sad[tid + 64];
                        s_mv[tid].y = s_mv[tid + 64].y;
                        s_mv[tid].x = s_mv[tid + 64].x;
                }
        }
#endif
        __syncthreads();
        
        if(tid < 32){   //in 1 warp, needless __syncthreads()  
                if(s_sad[tid] > s_sad[tid + 32]){
                        s_sad[tid] = s_sad[tid + 32];
                        s_mv[tid].y = s_mv[tid + 32].y;
                        s_mv[tid].x = s_mv[tid + 32].x;
                }
                if(s_sad[tid] > s_sad[tid + 16]){
                        s_sad[tid] = s_sad[tid + 16];
                        s_mv[tid].y = s_mv[tid + 16].y;
                        s_mv[tid].x = s_mv[tid + 16].x;
                }
                if(s_sad[tid] > s_sad[tid + 8]){
                        s_sad[tid] = s_sad[tid + 8];
                        s_mv[tid].y = s_mv[tid + 8].y;
                        s_mv[tid].x = s_mv[tid + 8].x;
                }
                if(s_sad[tid] > s_sad[tid + 4]){
                        s_sad[tid] = s_sad[tid + 4];
                        s_mv[tid].y = s_mv[tid + 4].y;
                        s_mv[tid].x = s_mv[tid + 4].x;
                }
                if(s_sad[tid] > s_sad[tid + 2]){
                        s_sad[tid] = s_sad[tid + 2];
                        s_mv[tid].y = s_mv[tid + 2].y;
                        s_mv[tid].x = s_mv[tid + 2].x;
                }
                if(s_sad[0] > s_sad[1]){
                        d_sad[by*(IMAGE_WIDTH/8) + bx] = s_sad[1];
                        d_mvY[by*(IMAGE_WIDTH/8) + bx] = s_mv[1].y;
                        d_mvX[by*(IMAGE_WIDTH/8) + bx] = s_mv[1].x;
                }else{
                        d_sad[by*(IMAGE_WIDTH/8) + bx] = s_sad[0];
                        d_mvY[by*(IMAGE_WIDTH/8) + bx] = s_mv[0].y;
                        d_mvX[by*(IMAGE_WIDTH/8) + bx] = s_mv[0].x;
                }
        }
}
//---------------------------------------------------------------------------------------------------
__global__ void kernelPmods16x8Mv( int orgStride, short* yOrg,
                                   short* d_mvX, short* d_mvY, uint* d_sad){

        uint tid = threadIdx.x;
        uint bx  = blockIdx.x;
        uint by  = blockIdx.y;
        
        short pucCurX = SEARCH_RANGE;
        short pucCurY = SEARCH_RANGE;
        short* pucOrg;
        
                        
        __shared__ short2 s_mv[BLOCK_DIM];
        __shared__ uint   s_sad[BLOCK_DIM];
        __shared__ short  s_orgMb[128];
        short mvY;
        uint  curSad;
            
        pucOrg = yOrg + Shift2Next16x8Block(orgStride, by, bx);
        pucCurX += bx*16;
        pucCurY += by*8;
        
        pucCurX += (tid - SEARCH_RANGE);
        
        short  pucCurY1;
        short  pucCurY2;
        short  pucCurY3;
        short  pucCurY4;
        short  pucCurY5;
        short  pucCurY6;
        
        uint    cur1Sad = 0;
        uint    cur2Sad = 0;
        uint    cur3Sad = 0;
        uint    cur4Sad = 0;
        uint    cur5Sad = 0;
        uint    cur6Sad = 0;
        

        mvY = 0;
        curSad = 0;
        
        pucCurY1  = pucCurY + P1;
        pucCurY2  = pucCurY + P2;
        pucCurY3  = pucCurY + P3;
        pucCurY4  = pucCurY + P4;
        pucCurY5  = pucCurY + P5;
        pucCurY6  = pucCurY + P6;

        if(tid < 128){
                s_orgMb[tid] = pucOrg[Index2DAddress(orgStride, tid>>4, tid&15)];
        }
        __syncthreads(); 
        
        for(int m = 0; m < 8; m++){
                for(int n = 0; n < 16; n++){
                        
                        curSad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY + m), s_orgMb[Index2DAddress(16, m, n)], curSad);
                        cur1Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(16, m, n)], cur1Sad);
                        cur2Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(16, m, n)], cur2Sad);
                        cur3Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY3 + m), s_orgMb[Index2DAddress(16, m, n)], cur3Sad);
                        cur4Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY4 + m), s_orgMb[Index2DAddress(16, m, n)], cur4Sad);
                        cur5Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY5 + m), s_orgMb[Index2DAddress(16, m, n)], cur5Sad);
                        cur6Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY6 + m), s_orgMb[Index2DAddress(16, m, n)], cur6Sad);
                }
        }

        
        short distance = 0;     
        if(curSad > cur1Sad) { distance = P1; curSad = cur1Sad;}
        if(curSad > cur2Sad) { distance = P2; curSad = cur2Sad;}
        if(curSad > cur3Sad) { distance = P3; curSad = cur3Sad;}
        if(curSad > cur4Sad) { distance = P4; curSad = cur4Sad;}
        if(curSad > cur5Sad) { distance = P5; curSad = cur5Sad;}
        if(curSad > cur6Sad) { distance = P6; curSad = cur6Sad;}
        
        mvY += distance;
        
        for(int s = STEP_SIZE; s > 0; s >>= 1){

               
                pucCurY = pucCurY + distance;
                cur1Sad = 0;
                cur2Sad = 0;
        
                pucCurY1  = pucCurY + (-s);
                pucCurY2  = pucCurY + s;
                
                for(int m = 0; m < 8; m++){
                        for(int n = 0; n < 16; n++){
                                cur1Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(16, m, n)], cur1Sad);
                                cur2Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(16, m, n)], cur2Sad);
                        }
                }

                distance = 0;  
                if(curSad > cur1Sad) { distance = -s; curSad = cur1Sad;}
                if(curSad > cur2Sad) { distance = s; curSad = cur2Sad;}
                mvY += distance;          

        }
        
        s_mv[tid].y = mvY;
        s_sad[tid] = curSad;

        s_mv[tid].x = tid - SEARCH_RANGE;
                                
        
#if SEARCH_RANGE > 32
        __syncthreads();

        if(tid < 64){
                if(s_sad[tid] > s_sad[tid + 64]){
                        s_sad[tid] = s_sad[tid + 64];
                        s_mv[tid].y = s_mv[tid + 64].y;
                        s_mv[tid].x = s_mv[tid + 64].x;
                }
        }
#endif
        __syncthreads();
        
        if(tid < 32){   //in 1 warp, needless __syncthreads()   
                if(s_sad[tid] > s_sad[tid + 32]){
                        s_sad[tid] = s_sad[tid + 32];
                        s_mv[tid].y = s_mv[tid + 32].y;
                        s_mv[tid].x = s_mv[tid + 32].x;
                }
                if(s_sad[tid] > s_sad[tid + 16]){
                        s_sad[tid] = s_sad[tid + 16];
                        s_mv[tid].y = s_mv[tid + 16].y;
                        s_mv[tid].x = s_mv[tid + 16].x;
                }
                if(s_sad[tid] > s_sad[tid + 8]){
                        s_sad[tid] = s_sad[tid + 8];
                        s_mv[tid].y = s_mv[tid + 8].y;
                        s_mv[tid].x = s_mv[tid + 8].x;
                }
                if(s_sad[tid] > s_sad[tid + 4]){
                        s_sad[tid] = s_sad[tid + 4];
                        s_mv[tid].y = s_mv[tid + 4].y;
                        s_mv[tid].x = s_mv[tid + 4].x;
                }
                if(s_sad[tid] > s_sad[tid + 2]){
                        s_sad[tid] = s_sad[tid + 2];
                        s_mv[tid].y = s_mv[tid + 2].y;
                        s_mv[tid].x = s_mv[tid + 2].x;
                }
                if(s_sad[0] > s_sad[1]){
                        d_sad[by*(IMAGE_WIDTH/16) + bx] = s_sad[1];
                        d_mvY[by*(IMAGE_WIDTH/16) + bx] = s_mv[1].y;
                        d_mvX[by*(IMAGE_WIDTH/16) + bx] = s_mv[1].x;
                }else{
                        d_sad[by*(IMAGE_WIDTH/16) + bx] = s_sad[0];
                        d_mvY[by*(IMAGE_WIDTH/16) + bx] = s_mv[0].y;
                        d_mvX[by*(IMAGE_WIDTH/16) + bx] = s_mv[0].x;
                }
        }
        
}
//---------------------------------------------------------------------------------------------------
__global__ void kernelPmods16x16Mv(int orgStride, short* yOrg,
                                   short* d_mvX, short* d_mvY, uint* d_sad){

        uint tid = threadIdx.x;
        uint bx  = blockIdx.x;
        uint by  = blockIdx.y;
        
        
        short pucCurX = SEARCH_RANGE;
        short pucCurY = SEARCH_RANGE;
        short* pucOrg;
        
                        
        __shared__ short2 s_mv[BLOCK_DIM];
        __shared__ uint   s_sad[BLOCK_DIM];
        __shared__ short  s_orgMb[256];
        short mvY;
        uint  curSad;
                          
        pucOrg = yOrg + Shift2Next16x16Block(orgStride, by, bx);

        pucCurX += bx*16;
        pucCurY += by*16;
        
        pucCurX += (tid - SEARCH_RANGE);

        curSad = 0;
    

        short  pucCurY1;
        short  pucCurY2;
        short  pucCurY3;
        short  pucCurY4;
        short  pucCurY5;
        short  pucCurY6;
        
        uint    cur1Sad = 0;
        uint    cur2Sad = 0;
        uint    cur3Sad = 0;
        uint    cur4Sad = 0;
        uint    cur5Sad = 0;
        uint    cur6Sad = 0;
        
        mvY = 0;
        curSad = 0;
        
        pucCurY1  = pucCurY + P1;
        pucCurY2  = pucCurY + P2;
        pucCurY3  = pucCurY + P3;
        pucCurY4  = pucCurY + P4;
        pucCurY5  = pucCurY + P5;
        pucCurY6  = pucCurY + P6;

        
        s_orgMb[tid] = pucOrg[Index2DAddress(orgStride, tid>>4, tid&15)];
        s_orgMb[tid + 128] = pucOrg[Index2DAddress(orgStride, 8 + (tid>>4), tid&15)];
        __syncthreads(); 
        
        for(int m = 0; m < 16; m++){
                for(int n = 0; n < 16; n++){
                        
                        curSad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY + m), s_orgMb[Index2DAddress(16, m, n)], curSad);
                        cur1Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(16, m, n)], cur1Sad);
                        cur2Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(16, m, n)], cur2Sad);
                        cur3Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY3 + m), s_orgMb[Index2DAddress(16, m, n)], cur3Sad);
                        cur4Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY4 + m), s_orgMb[Index2DAddress(16, m, n)], cur4Sad);
                        cur5Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY5 + m), s_orgMb[Index2DAddress(16, m, n)], cur5Sad);
                        cur6Sad = __usad(tex2D(tex_ref, pucCurX + n, pucCurY6 + m), s_orgMb[Index2DAddress(16, m, n)], cur6Sad);
                }
        }

        
        short distance = 0;   
        if(curSad > cur1Sad) { distance = P1; curSad = cur1Sad;}
        if(curSad > cur2Sad) { distance = P2; curSad = cur2Sad;}
        if(curSad > cur3Sad) { distance = P3; curSad = cur3Sad;}
        if(curSad > cur4Sad) { distance = P4; curSad = cur4Sad;}
        if(curSad > cur5Sad) { distance = P5; curSad = cur5Sad;}
        if(curSad > cur6Sad) { distance = P6; curSad = cur6Sad;}
        
        mvY += distance;

       
        for(int s = STEP_SIZE; s > 0; s >>= 1){

               
                pucCurY = pucCurY + distance;
                cur1Sad = 0;
                cur2Sad = 0;
        
                pucCurY1  = pucCurY + (-s);
                pucCurY2  = pucCurY + s;

               
                for(int m = 0; m < 16; m++){
                        for(int n = 0; n < 16; n++){
                                
                                cur1Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY1 + m), s_orgMb[Index2DAddress(16, m, n)], cur1Sad);
                                cur2Sad  = __usad(tex2D(tex_ref, pucCurX + n, pucCurY2 + m), s_orgMb[Index2DAddress(16, m, n)], cur2Sad);
                                
                        }
                }

                distance = 0;   
                if(curSad > cur1Sad) { distance = -s; curSad = cur1Sad;}
                if(curSad > cur2Sad) { distance = s; curSad = cur2Sad;}
                mvY += distance;          

        }
        
        s_mv[tid].y = mvY;
        s_sad[tid] = curSad;

        s_mv[tid].x = tid - SEARCH_RANGE;
                                
                        
#if SEARCH_RANGE > 32
        __syncthreads();

        if(tid < 64){
                if(s_sad[tid] > s_sad[tid + 64]){
                        s_sad[tid] = s_sad[tid + 64];
                        s_mv[tid].y = s_mv[tid + 64].y;
                        s_mv[tid].x = s_mv[tid + 64].x;
                }
        }
#endif
        __syncthreads();
        
        if(tid < 32){   //in 1 warp, needless __syncthreads()   
                if(s_sad[tid] > s_sad[tid + 32]){
                        s_sad[tid] = s_sad[tid + 32];
                        s_mv[tid].y = s_mv[tid + 32].y;
                        s_mv[tid].x = s_mv[tid + 32].x;
                }
                if(s_sad[tid] > s_sad[tid + 16]){
                        s_sad[tid] = s_sad[tid + 16];
                        s_mv[tid].y = s_mv[tid + 16].y;
                        s_mv[tid].x = s_mv[tid + 16].x;
                }
                if(s_sad[tid] > s_sad[tid + 8]){
                        s_sad[tid] = s_sad[tid + 8];
                        s_mv[tid].y = s_mv[tid + 8].y;
                        s_mv[tid].x = s_mv[tid + 8].x;
                }
                if(s_sad[tid] > s_sad[tid + 4]){
                        s_sad[tid] = s_sad[tid + 4];
                        s_mv[tid].y = s_mv[tid + 4].y;
                        s_mv[tid].x = s_mv[tid + 4].x;
                }
                if(s_sad[tid] > s_sad[tid + 2]){
                        s_sad[tid] = s_sad[tid + 2];
                        s_mv[tid].y = s_mv[tid + 2].y;
                        s_mv[tid].x = s_mv[tid + 2].x;
                }
                if(s_sad[0] > s_sad[1]){
                        d_sad[by*(IMAGE_WIDTH/16) + bx] = s_sad[1];
                        d_mvY[by*(IMAGE_WIDTH/16) + bx] = s_mv[1].y;
                        d_mvX[by*(IMAGE_WIDTH/16) + bx] = s_mv[1].x;
                }else{
                        d_sad[by*(IMAGE_WIDTH/16) + bx] = s_sad[0];
                        d_mvY[by*(IMAGE_WIDTH/16) + bx] = s_mv[0].y;
                        d_mvX[by*(IMAGE_WIDTH/16) + bx] = s_mv[0].x;
                }
        }
}
//---------------------------------------------------------------------------------------------------
__global__ void kerneltest(int refStride, short* yRef, short* result1, short* result2){
        uint tid = threadIdx.x;
        uint bx  = blockIdx.x;
        uint by  = blockIdx.y;
        
        short* pucCur = yRef;
        short pucCurX = SEARCH_RANGE;
        short pucCurY = SEARCH_RANGE;
        
      
        //pucOrg = yOrg + Shift2Next16x16Block(orgStride, by, bx);
        //pucCur = yRef + Shift2Next16x16Block(refStride, by, bx);
        //pucCurX += bx*16;
        //pucCurY += by*16;
        pucCur += (-SEARCH_RANGE);
        pucCurX += (-SEARCH_RANGE);

#if 1    
        if(bx == 0 && by == 0){
                if(tid == 0){
                        for(int i = 0; i < 128; i++){
                                result1[i] = tex2D(tex_ref, pucCurX +i, pucCurY);
                                result2[i] = pucCur[Index2DAddress(refStride, 0, i)];
                        }
                }
        }
#endif
}
//---------------------------------------------------------------------------------------------------
void eggCudaPmodsMalloc(){

        int refStride = externEggEstimation.m_refStride;
       
        pucOrgStart = externEggEstimation.m_pYOrg;                                        
        pucRefStart = externEggEstimation.m_pYRef - Index2DAddress( refStride,
                                                                    SEARCH_RANGE,
                                                                    SEARCH_RANGE);
                                                
        refImageSize = (IMAGE_HEIGHT + SEARCH_RANGE*2)*
                       (IMAGE_WIDTH  + SEARCH_RANGE)*sizeof(short);
        
        orgImageSize = (IMAGE_HEIGHT + SEARCH_RANGE)*(IMAGE_WIDTH + SEARCH_RANGE)*sizeof(short);
        
        
        cutilSafeCall( cudaMalloc( (void**) &d_yOrg, orgImageSize));
        
        cutilSafeCall( cudaMemcpy( d_yOrg, pucOrgStart, orgImageSize, cudaMemcpyHostToDevice) );

        cutilSafeCall( cudaMallocArray( &cu_array_ref, &channelDesc, 
                                        (refStride),
                                        (IMAGE_HEIGHT + SEARCH_RANGE*2) ) );

          
        cutilSafeCall( cudaMemcpyToArray(   cu_array_ref, 0, 0, pucRefStart, 
                                            refImageSize, cudaMemcpyHostToDevice ) );

        cutilSafeCall( cudaBindTextureToArray( tex_ref, cu_array_ref ));

        
}

//---------------------------------------------------------------------------------------------------
void eggCudaPmodsFree(){

        cutilSafeCall( cudaFree(d_yOrg) );
        cutilSafeCall( cudaUnbindTexture(tex_ref) );
        cutilSafeCall( cudaFreeArray(cu_array_ref) );
}
//---------------------------------------------------------------------------------------------------
//parallel multithread one-dimensional search
void eggCudaPmods4x4Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock){
       
        short*  h_mvX = new short[(IMAGE_WIDTH/4)*(IMAGE_HEIGHT/4)];
        short*  h_mvY = new short[(IMAGE_WIDTH/4)*(IMAGE_HEIGHT/4)];
        uint*   h_sad = new uint[(IMAGE_WIDTH/4)*(IMAGE_HEIGHT/4)];
        
        short*  d_mvX;
        short*  d_mvY;
        uint*   d_sad;

        cutilSafeCall(cudaMalloc((void**)&d_mvX,(IMAGE_WIDTH/4)*(IMAGE_HEIGHT/4)*sizeof(short)));
        cutilSafeCall(cudaMalloc((void**)&d_mvY,(IMAGE_WIDTH/4)*(IMAGE_HEIGHT/4)*sizeof(short)));
        cutilSafeCall(cudaMalloc((void**)&d_sad,(IMAGE_WIDTH/4)*(IMAGE_HEIGHT/4)*sizeof(uint)));

        //cudaEvent_t start, stop;
        //cutilSafeCall(cudaEventCreate(&start));
        //cutilSafeCall(cudaEventCreate(&stop));
        
        //cudaEventRecord(start, 0);                                        
        kernelPmods4x4Mv<<<grid4x4Mv, BLOCK_DIM>>>( externEggEstimation.m_orgStride,
                                                    d_yOrg, d_mvX, d_mvY, d_sad);
        // check for any errors
        cutilCheckMsg("kernelPmods4x4Mv execution failed");
        //cudaEventRecord(stop, 0);
/*
        while(cudaEventQuery(stop) == cudaErrorNotReady){
                
        }
*/
        
        cutilSafeCall( cudaMemcpy( h_mvX, d_mvX, (IMAGE_WIDTH/4)*(IMAGE_HEIGHT/4)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_mvY, d_mvY, (IMAGE_WIDTH/4)*(IMAGE_HEIGHT/4)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_sad, d_sad, (IMAGE_WIDTH/4)*(IMAGE_HEIGHT/4)*sizeof(uint),
                                   cudaMemcpyDeviceToHost) );

        for(int i = 0; i < IMAGE_HEIGHT/4; i++){
        	for(int j = 0; j < IMAGE_WIDTH/4; j++){
                        r2ListMv.mv4x4[i][j].x   = h_mvX[i*(IMAGE_WIDTH/4) + j];
                        r2ListMv.mv4x4[i][j].y   = h_mvY[i*(IMAGE_WIDTH/4) + j];
                        r2ListBlock.sad4x4[i][j] = h_sad[i*(IMAGE_WIDTH/4) + j];
                }
        }
        
        cutilSafeCall(cudaFree(d_mvX));
        cutilSafeCall(cudaFree(d_mvY));
        cutilSafeCall(cudaFree(d_sad));

        delete[] h_mvX;
        delete[] h_mvY;
        delete[] h_sad;
        
}
//---------------------------------------------------------------------------------------------------
//parallel multithread one-dimensional search
void eggCudaPmods4x8Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock){
                                               
        
        short*  h_mvX = new short[(IMAGE_WIDTH/4)*(IMAGE_HEIGHT/8)];
        short*  h_mvY = new short[(IMAGE_WIDTH/4)*(IMAGE_HEIGHT/8)];
        uint*   h_sad = new uint[(IMAGE_WIDTH/4)*(IMAGE_HEIGHT/8)];
        short*  d_mvX;
        short*  d_mvY;
        uint*   d_sad;

        cutilSafeCall(cudaMalloc((void**)&d_mvX,(IMAGE_WIDTH/4)*(IMAGE_HEIGHT/8)*sizeof(short)));
        cutilSafeCall(cudaMalloc((void**)&d_mvY,(IMAGE_WIDTH/4)*(IMAGE_HEIGHT/8)*sizeof(short)));
        cutilSafeCall(cudaMalloc((void**)&d_sad,(IMAGE_WIDTH/4)*(IMAGE_HEIGHT/8)*sizeof(uint)));

                                             
        kernelPmods4x8Mv<<<grid4x8Mv, BLOCK_DIM>>>( externEggEstimation.m_orgStride,
                                                    d_yOrg, d_mvX, d_mvY, d_sad);
        // check for any errors
        cutilCheckMsg("kernelPmods4x8Mv execution failed");
        
        cutilSafeCall( cudaMemcpy( h_mvX, d_mvX, (IMAGE_WIDTH/4)*(IMAGE_HEIGHT/8)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_mvY, d_mvY, (IMAGE_WIDTH/4)*(IMAGE_HEIGHT/8)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_sad, d_sad, (IMAGE_WIDTH/4)*(IMAGE_HEIGHT/8)*sizeof(uint),
                                   cudaMemcpyDeviceToHost) );

        for(int i = 0; i < IMAGE_HEIGHT/8; i++){
        	for(int j = 0; j < IMAGE_WIDTH/4; j++){
                        r2ListMv.mv4x8[i][j].x   = h_mvX[i*(IMAGE_WIDTH/4) + j];
                        r2ListMv.mv4x8[i][j].y   = h_mvY[i*(IMAGE_WIDTH/4) + j];
                        r2ListBlock.sad4x8[i][j] = h_sad[i*(IMAGE_WIDTH/4) + j];
                }
        }
        
        cutilSafeCall(cudaFree(d_mvX));
        cutilSafeCall(cudaFree(d_mvY));
        cutilSafeCall(cudaFree(d_sad));

        delete[] h_mvX;
        delete[] h_mvY;
        delete[] h_sad;
        
}
//---------------------------------------------------------------------------------------------------
//parallel multithread one-dimensional search
void eggCudaPmods8x4Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock){
                                               
        
        short*  h_mvX = new short[(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/4)];
        short*  h_mvY = new short[(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/4)];
        uint*   h_sad = new uint[(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/4)];
        short*  d_mvX;
        short*  d_mvY;
        uint*   d_sad;

        cutilSafeCall(cudaMalloc((void**)&d_mvX,(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/4)*sizeof(short)));
        cutilSafeCall(cudaMalloc((void**)&d_mvY,(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/4)*sizeof(short)));
        cutilSafeCall(cudaMalloc((void**)&d_sad,(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/4)*sizeof(uint)));

                                           
        kernelPmods8x4Mv<<<grid8x4Mv, BLOCK_DIM>>>( externEggEstimation.m_orgStride,
                                                    d_yOrg, d_mvX, d_mvY, d_sad);
        // check for any errors
        cutilCheckMsg("kernelPmods8x4Mv execution failed");
        
        
        cutilSafeCall( cudaMemcpy( h_mvX, d_mvX, (IMAGE_WIDTH/8)*(IMAGE_HEIGHT/4)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_mvY, d_mvY, (IMAGE_WIDTH/8)*(IMAGE_HEIGHT/4)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_sad, d_sad, (IMAGE_WIDTH/8)*(IMAGE_HEIGHT/4)*sizeof(uint),
                                   cudaMemcpyDeviceToHost) );

        for(int i = 0; i < IMAGE_HEIGHT/4; i++){
        	for(int j = 0; j < IMAGE_WIDTH/8; j++){
                        r2ListMv.mv8x4[i][j].x   = h_mvX[i*(IMAGE_WIDTH/8) + j];
                        r2ListMv.mv8x4[i][j].y   = h_mvY[i*(IMAGE_WIDTH/8) + j];
                        r2ListBlock.sad8x4[i][j] = h_sad[i*(IMAGE_WIDTH/8) + j];
                }
        }
        
        cutilSafeCall(cudaFree(d_mvX));
        cutilSafeCall(cudaFree(d_mvY));
        cutilSafeCall(cudaFree(d_sad));

        delete[] h_mvX;
        delete[] h_mvY;
        delete[] h_sad;
        
}
//---------------------------------------------------------------------------------------------------
//parallel multithread one-dimensional search
void eggCudaPmods8x8Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock){
                                                
        
        short*  h_mvX = new short[(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/8)];
        short*  h_mvY = new short[(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/8)];
        uint*   h_sad = new uint[(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/8)];
        short*  d_mvX;
        short*  d_mvY;
        uint*   d_sad;

        cutilSafeCall(cudaMalloc((void**)&d_mvX,(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/8)*sizeof(short)));
        cutilSafeCall(cudaMalloc((void**)&d_mvY,(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/8)*sizeof(short)));
        cutilSafeCall(cudaMalloc((void**)&d_sad,(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/8)*sizeof(uint)));

                                               
        kernelPmods8x8Mv<<<grid8x8Mv, BLOCK_DIM>>>( externEggEstimation.m_orgStride,
                                                    d_yOrg, d_mvX, d_mvY, d_sad);
        // check for any errors
        cutilCheckMsg("kernelPmods8x8Mv execution failed");
        
        
        cutilSafeCall( cudaMemcpy( h_mvX, d_mvX, (IMAGE_WIDTH/8)*(IMAGE_HEIGHT/8)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_mvY, d_mvY, (IMAGE_WIDTH/8)*(IMAGE_HEIGHT/8)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_sad, d_sad, (IMAGE_WIDTH/8)*(IMAGE_HEIGHT/8)*sizeof(uint),
                                   cudaMemcpyDeviceToHost) );

        for(int i = 0; i < IMAGE_HEIGHT/8; i++){
        	for(int j = 0; j < IMAGE_WIDTH/8; j++){
                        r2ListMv.mv8x8[i][j].x   = h_mvX[i*(IMAGE_WIDTH/8) + j];
                        r2ListMv.mv8x8[i][j].y   = h_mvY[i*(IMAGE_WIDTH/8) + j];
                        r2ListBlock.sad8x8[i][j] = h_sad[i*(IMAGE_WIDTH/8) + j];
                }
        }
        
        cutilSafeCall(cudaFree(d_mvX));
        cutilSafeCall(cudaFree(d_mvY));
        cutilSafeCall(cudaFree(d_sad));

        delete[] h_mvX;
        delete[] h_mvY;
        delete[] h_sad;
        
}
//---------------------------------------------------------------------------------------------------
//parallel multithread one-dimensional search
void eggCudaPmods8x16Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock){
                                               
        
        short*  h_mvX = new short[(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/16)];
        short*  h_mvY = new short[(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/16)];
        uint*   h_sad = new uint[(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/16)];
        short*  d_mvX;
        short*  d_mvY;
        uint*   d_sad;

        cutilSafeCall(cudaMalloc((void**)&d_mvX,(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/16)*sizeof(short)));
        cutilSafeCall(cudaMalloc((void**)&d_mvY,(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/16)*sizeof(short)));
        cutilSafeCall(cudaMalloc((void**)&d_sad,(IMAGE_WIDTH/8)*(IMAGE_HEIGHT/16)*sizeof(uint)));

                                              
        kernelPmods8x16Mv<<<grid8x16Mv, BLOCK_DIM>>>( externEggEstimation.m_orgStride,
                                                      d_yOrg, d_mvX, d_mvY, d_sad);
        // check for any errors
        cutilCheckMsg("kernelPmods8x16Mv execution failed");
        
        
        cutilSafeCall( cudaMemcpy( h_mvX, d_mvX, (IMAGE_WIDTH/8)*(IMAGE_HEIGHT/16)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_mvY, d_mvY, (IMAGE_WIDTH/8)*(IMAGE_HEIGHT/16)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_sad, d_sad, (IMAGE_WIDTH/8)*(IMAGE_HEIGHT/16)*sizeof(uint),
                                   cudaMemcpyDeviceToHost) );

        for(int i = 0; i < IMAGE_HEIGHT/16; i++){
        	for(int j = 0; j < IMAGE_WIDTH/8; j++){
                        r2ListMv.mv8x16[i][j].x   = h_mvX[i*(IMAGE_WIDTH/8) + j];
                        r2ListMv.mv8x16[i][j].y   = h_mvY[i*(IMAGE_WIDTH/8) + j];
                        r2ListBlock.sad8x16[i][j] = h_sad[i*(IMAGE_WIDTH/8) + j];
                }
        }
        
        cutilSafeCall(cudaFree(d_mvX));
        cutilSafeCall(cudaFree(d_mvY));
        cutilSafeCall(cudaFree(d_sad));

        delete[] h_mvX;
        delete[] h_mvY;
        delete[] h_sad;
        
}
//---------------------------------------------------------------------------------------------------
//parallel multithread one-dimensional search
void eggCudaPmods16x8Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock){
       
        
        short*  h_mvX = new short[(IMAGE_WIDTH/16)*(IMAGE_HEIGHT/8)];
        short*  h_mvY = new short[(IMAGE_WIDTH/16)*(IMAGE_HEIGHT/8)];
        uint*   h_sad = new uint[(IMAGE_WIDTH/16)*(IMAGE_HEIGHT/8)];
        short*  d_mvX;
        short*  d_mvY;
        uint*   d_sad;

        cutilSafeCall(cudaMalloc((void**)&d_mvX,(IMAGE_WIDTH/16)*(IMAGE_HEIGHT/8)*sizeof(short)));
        cutilSafeCall(cudaMalloc((void**)&d_mvY,(IMAGE_WIDTH/16)*(IMAGE_HEIGHT/8)*sizeof(short)));
        cutilSafeCall(cudaMalloc((void**)&d_sad,(IMAGE_WIDTH/16)*(IMAGE_HEIGHT/8)*sizeof(uint)));

                                               
        kernelPmods16x8Mv<<<grid16x8Mv, BLOCK_DIM>>>( externEggEstimation.m_orgStride,
                                                      d_yOrg, d_mvX, d_mvY, d_sad);
        // check for any errors
        cutilCheckMsg("kernelPmods16x8Mv execution failed");
        
        
        cutilSafeCall( cudaMemcpy( h_mvX, d_mvX, (IMAGE_WIDTH/16)*(IMAGE_HEIGHT/8)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_mvY, d_mvY, (IMAGE_WIDTH/16)*(IMAGE_HEIGHT/8)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_sad, d_sad, (IMAGE_WIDTH/16)*(IMAGE_HEIGHT/8)*sizeof(uint),
                                   cudaMemcpyDeviceToHost) );

        for(int i = 0; i < IMAGE_HEIGHT/8; i++){
        	for(int j = 0; j < IMAGE_WIDTH/16; j++){
                        r2ListMv.mv16x8[i][j].x   = h_mvX[i*(IMAGE_WIDTH/16) + j];
                        r2ListMv.mv16x8[i][j].y   = h_mvY[i*(IMAGE_WIDTH/16) + j];
                        r2ListBlock.sad16x8[i][j] = h_sad[i*(IMAGE_WIDTH/16) + j];
                }
        }
        
        cutilSafeCall(cudaFree(d_mvX));
        cutilSafeCall(cudaFree(d_mvY));
        cutilSafeCall(cudaFree(d_sad));

        delete[] h_mvX;
        delete[] h_mvY;
        delete[] h_sad;
}

//---------------------------------------------------------------------------------------------------
//parallel multithread one-dimensional search
void eggCudaPmods16x16Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock){
                                              
        short*  h_mvX = new short[(IMAGE_WIDTH/16)*(IMAGE_HEIGHT/16)];
        short*  h_mvY = new short[(IMAGE_WIDTH/16)*(IMAGE_HEIGHT/16)];
        uint*   h_sad = new uint[(IMAGE_WIDTH/16)*(IMAGE_HEIGHT/16)];
        
        short*  d_mvX;
        short*  d_mvY;
        uint*   d_sad;
        
        
        cutilSafeCall( cudaMalloc((void**)&d_mvX,(IMAGE_WIDTH/16)*(IMAGE_HEIGHT/16)*sizeof(short)));
        cutilSafeCall( cudaMalloc((void**)&d_mvY,(IMAGE_WIDTH/16)*(IMAGE_HEIGHT/16)*sizeof(short)));
        cutilSafeCall( cudaMalloc((void**)&d_sad,(IMAGE_WIDTH/16)*(IMAGE_HEIGHT/16)*sizeof(uint)));
        

        kernelPmods16x16Mv<<<grid16x16Mv, BLOCK_DIM>>>( externEggEstimation.m_orgStride,
                                                        d_yOrg, d_mvX, d_mvY, d_sad);
        // check for any errors
        cutilCheckMsg("kernelPmods16x16Mv execution failed");

        
        cutilSafeCall( cudaMemcpy( h_mvX, d_mvX, (IMAGE_WIDTH/16)*(IMAGE_HEIGHT/16)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
       
        cutilSafeCall( cudaMemcpy( h_mvY, d_mvY, (IMAGE_WIDTH/16)*(IMAGE_HEIGHT/16)*sizeof(short),
                                   cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy( h_sad, d_sad, (IMAGE_WIDTH/16)*(IMAGE_HEIGHT/16)*sizeof(uint),
                                   cudaMemcpyDeviceToHost) );
        
        
        for(int i = 0; i < IMAGE_HEIGHT/16; i++){
        	for(int j = 0; j < IMAGE_WIDTH/16; j++){
                        r2ListMv.mv16x16[i][j].x   = h_mvX[i*(IMAGE_WIDTH/16) + j];
                        r2ListMv.mv16x16[i][j].y   = h_mvY[i*(IMAGE_WIDTH/16) + j];
                        r2ListBlock.sad16x16[i][j] = h_sad[i*(IMAGE_WIDTH/16) + j];
                }
        }

        
        cutilSafeCall(cudaFree(d_mvX));
        cutilSafeCall(cudaFree(d_mvY));
        cutilSafeCall(cudaFree(d_sad));
        
        delete[] h_mvX;
        delete[] h_mvY;
        delete[] h_sad;
}

