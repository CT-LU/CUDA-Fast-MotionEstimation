#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <sys/time.h>

using namespace std;

//---------------------------------------------------------------------------------------------------
enum LogMbMode
{
  MODE_SKIP         = 0,
  MODE_16x16        = 1,
  MODE_16x8         = 2,
  MODE_8x16         = 3,
  MODE_8x8          = 4,
  MODE_8x8ref0      = 5,
  INTRA_4X4         = 6,
  MODE_PCM          = 25+6,
  INTRA_BL          = 36
};

enum LogBlkMode
{
  BLK_8x8   = 8,
  BLK_8x4   = 9,
  BLK_4x8   = 10,
  BLK_4x4   = 11,
  BLK_SKIP  = 0
};


//---------------------------------------------------------------------------------------------------
#define _EGG_ESTIMATION_ENTRY_ 0		// 0 : original jmvc || 1 : proposed algo
#define _IS_MVP_NOTUSED  0			// 0 : turn on mvp  || 1 : turn off mvp 
#define _DISABLE_RDO     0			// disable fast search

#define _LOG_MV 0				// keep all motion vectors for debugging 
#define _LOG_MV_MODE MODE_16x16			// determine what kind of mode to save for debugging


#define SEARCH_RANGE	64			
#define Y_SEARCH_RANGE  12 
#define INIT_STEP	4

#define STEP_SIZE	2	
#define P1		-12			
#define P2		-8			
#define P3		-4			
#define P4		4
#define P5		8	
#define P6		12			

#define BLOCK_DIM	128		
#define BORDERSIZE	128			// SEARCH_RANGE*4
#define INTERPOLATEDIM	128
#define IMAGE_HEIGHT	480
#define IMAGE_WIDTH	640

//---------------------------------------------------------------------------------------------------
#define MSYS_UINT_MAX		      0xFFFFFFFFU

#define Shift2Next4x4Block(stride, i, j) ((i)*(stride)*4 + (j)*4)
#define Shift2Next4x8Block(stride, i, j) ((i)*(stride)*8 + (j)*4)
#define Shift2Next8x4Block(stride, i, j) ((i)*(stride)*4 + (j)*8)
#define Shift2Next8x8Block(stride, i, j) ((i)*(stride)*8 + (j)*8)
#define Shift2Next8x16Block(stride, i, j) ((i)*(stride)*16 + (j)*8)
#define Shift2Next16x8Block(stride, i, j) ((i)*(stride)*8 + (j)*16)
#define Shift2Next16x16Block(stride, i, j) ((i)*(stride)*16 + (j)*16)

#define Index2DAddress(stride, m, n) ((m)*(stride) + (n))
//---------------------------------------------------------------------------------------------------

typedef unsigned int uint;
typedef unsigned short ushort;
typedef enum { LIST0, LIST1, BIDIRECT }List_t;
typedef enum { ME, DE }Estimation_t;
enum ParIdx16x8
{
  PART_16x8_0   = 0x00,
  PART_16x8_1   = 0x08
};
enum ParIdx8x16
{
  PART_8x16_0   = 0x00,
  PART_8x16_1   = 0x02
};
enum Par8x8
{
  B_8x8_0    = 0x00,
  B_8x8_1    = 0x01,
  B_8x8_2    = 0x02,
  B_8x8_3    = 0x03
};
enum ParIdx8x8
{
  PART_8x8_0    = 0x00,
  PART_8x8_1    = 0x02,
  PART_8x8_2    = 0x08,
  PART_8x8_3    = 0x0A
};
enum SParIdx8x8
{
  SPART_8x8   = 0x00
};
enum SParIdx8x4
{
  SPART_8x4_0   = 0x00,
  SPART_8x4_1   = 0x04
};
enum SParIdx4x8
{
  SPART_4x8_0   = 0x00,
  SPART_4x8_1   = 0x01
};
enum SParIdx4x4
{
  SPART_4x4_0   = 0x00,
  SPART_4x4_1   = 0x01,
  SPART_4x4_2   = 0x04,
  SPART_4x4_3   = 0x05
};

//---------------------------------------------------------------------------------------------------
typedef struct mvStruct{
	short x;
	short y;
}mv_t;

typedef struct mvModeStruct{
	mv_t mv4x4[IMAGE_HEIGHT/4][IMAGE_WIDTH/4];
	mv_t mv4x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/4];
	mv_t mv8x4[IMAGE_HEIGHT/4][IMAGE_WIDTH/8];
	mv_t mv8x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/8];
	mv_t mv8x16[IMAGE_HEIGHT/16][IMAGE_WIDTH/8];
	mv_t mv16x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/16];
	mv_t mv16x16[IMAGE_HEIGHT/16][IMAGE_WIDTH/16];

	mv_t mvp4x4[IMAGE_HEIGHT/4][IMAGE_WIDTH/4];
	mv_t mvp4x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/4];
	mv_t mvp8x4[IMAGE_HEIGHT/4][IMAGE_WIDTH/8];
	mv_t mvp8x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/8];
	mv_t mvp8x16[IMAGE_HEIGHT/16][IMAGE_WIDTH/8];
	mv_t mvp16x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/16];
	mv_t mvp16x16[IMAGE_HEIGHT/16][IMAGE_WIDTH/16];

	mv_t fmv4x4[IMAGE_HEIGHT/4][IMAGE_WIDTH/4];
	mv_t fmv4x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/4];
	mv_t fmv8x4[IMAGE_HEIGHT/4][IMAGE_WIDTH/8];
	mv_t fmv8x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/8];
	mv_t fmv8x16[IMAGE_HEIGHT/16][IMAGE_WIDTH/8];
	mv_t fmv16x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/16];
	mv_t fmv16x16[IMAGE_HEIGHT/16][IMAGE_WIDTH/16];
}mvMode_t;

typedef struct blockModeStruct{
	unsigned short 	sad16x16[IMAGE_HEIGHT/16][IMAGE_WIDTH/16];
	unsigned short 	sad16x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/16];
	unsigned short 	sad8x16[IMAGE_HEIGHT/16][IMAGE_WIDTH/8];
	unsigned short 	sad8x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/8];
	unsigned short 	sad8x4[IMAGE_HEIGHT/4][IMAGE_WIDTH/8];
	unsigned short 	sad4x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/4];
	unsigned short 	sad4x4[IMAGE_HEIGHT/4][IMAGE_WIDTH/4];

	unsigned short 	f_sad16x16[IMAGE_HEIGHT/16][IMAGE_WIDTH/16];
	unsigned short 	f_sad16x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/16];
	unsigned short 	f_sad8x16[IMAGE_HEIGHT/16][IMAGE_WIDTH/8];
	unsigned short 	f_sad8x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/8];
	unsigned short 	f_sad8x4[IMAGE_HEIGHT/4][IMAGE_WIDTH/8];
	unsigned short 	f_sad4x8[IMAGE_HEIGHT/8][IMAGE_WIDTH/4];
	unsigned short 	f_sad4x4[IMAGE_HEIGHT/4][IMAGE_WIDTH/4];

}blockMode_t;

class SearchTable{
public:
	short getX(int k){ return index2x[k]; }
	short getY(int k){ return index2y[k]; }
private:
	static short index2x[24];	// 24 point around the central point
	static short index2y[24];
};
//---------------------------------------------------------------------------------------------------
extern "C" void eggCudaPmodsMalloc();
extern "C" void eggCudaPmodsFree();
extern "C" void eggCudaPmods4x4Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
extern "C" void eggCudaPmods4x8Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
extern "C" void eggCudaPmods8x4Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
extern "C" void eggCudaPmods8x8Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
extern "C" void eggCudaPmods8x16Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
extern "C" void eggCudaPmods16x8Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
extern "C" void eggCudaPmods16x16Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
extern "C" void eggCudaInterpolateHpel();
extern "C" void eggCudaInterpolateQpel();
//---------------------------------------------------------------------------------------------------
class EggMotionEstimation{
public:	
	
	short			m_subPelRef[(IMAGE_HEIGHT + SEARCH_RANGE*2)*4][(IMAGE_WIDTH + SEARCH_RANGE*2)*4];
	mvMode_t		m_list0MeMv;
	mvMode_t		m_list1MeMv;
	mvMode_t		m_list0DeMv;
	mvMode_t		m_list1DeMv;

	blockMode_t		m_list0MeBlock;
	blockMode_t		m_list1MeBlock;
	blockMode_t		m_list0DeBlock;
	blockMode_t		m_list1DeBlock;

	SearchTable		m_srchTable;
	short*          	m_pYOrg;
	short*          	m_pYRef;
	int             	m_orgStride;
	int             	m_refStride;
	uint			m_uiMbY;
	uint			m_uiMbX;
	List_t			m_lstType;
	Estimation_t		m_isMEorDE;
	
	void		eggSetPmods4x4Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggSetPmods4x8Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggSetPmods8x4Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggSetPmods8x8Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggSetPmods8x16Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggSetPmods16x8Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggSetPmods16x16Mv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);

	void		eggInterpolateSubPel();
	void		eggSet4x4fMv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggSet4x8fMv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggSet8x4fMv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggSet8x8fMv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggSet8x16fMv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggSet16x8fMv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggSet16x16fMv(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	
	void		eggGet4x4Mv(short& sHor, short& sVer, uint& uiMinSAD, uint uiBlk);
	void		eggGet4x8Mv(short& sHor, short& sVer, uint& uiMinSAD, uint uiBlk);
	void		eggGet8x4Mv(short& sHor, short& sVer, uint& uiMinSAD, uint uiBlk);
	void		eggGet8x8Mv(short& sHor, short& sVer, uint& uiMinSAD, uint uiBlk);
	void		eggGet8x16Mv(short& sHor, short& sVer, uint& uiMinSAD, uint uiBlk);
	void		eggGet16x8Mv(short& sHor, short& sVer, uint& uiMinSAD, uint uiBlk);
	void		eggGet16x16Mv(short& sHor, short& sVer, uint& uiMinSAD);
	void		eggCudaIntEstimation(mvMode_t &r2ListMv, blockMode_t &r2ListBlock);
	void		eggCudaInterpolateSubPel();
	void		eggCudaMalloc();
	void		eggCudaFree();
};	

extern EggMotionEstimation externEggEstimation;	

