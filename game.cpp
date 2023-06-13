#include "precomp.h"
#include "game.h"

#define GRIDSIZE 256

// VERLET CLOTH SIMULATION DEMO
// High-level concept: a grid consists of points, each connected to four 
// neighbours. For a simulation step, the position of each point is affected
// by its speed, expressed as (current position - previous position), a
// constant gravity force downwards, and random impulses ("wind").
// The final force is provided by the bonds between points, via the four
// connections.
// Together, this simple scheme yields a pretty convincing cloth simulation.
// The algorithm has been used in games since the game "Thief".

// ASSIGNMENT STEPS:
// 1. SIMD, part 1: in Game::Simulation, convert lines 119 to 126 to SIMD.
//    You receive 2 points if the resulting code is faster than the original.
//    This will probably require a reorganization of the data layout, which
//    may in turn require changes to the rest of the code.
// 2. SIMD, part 2: for an additional 4 points, convert the full Simulation
//    function to SSE. This may require additional changes to the data to
//    avoid concurrency issues when operating on neighbouring points.
//    The resulting code must be at least 2 times faster (using SSE) or 4
//    times faster (using AVX) than the original  to receive the full 4 points.
// 3. GPGPU, part 1: modify Game::Simulation so that it sends the cloth data
//    to the GPU, and execute lines 119 to 126 on the GPU. After this, bring
//    back the cloth data to the CPU and execute the remainder of the Verlet
//    simulation code. You receive 2 points if the code *works* correctly;
//    note that this is expected to be slower due to the data transfers.
// 4. GPGPU, part 2: execute the full Game::Simulation function on the GPU.
//    You receive 4 additional points if this yields a correct simulation
//    that is at least 5x faster than the original code. DO NOT draw the
//    cloth on the GPU; this is (for now) outside the scope of the assignment.
// Note that the GPGPU tasks will benefit from the SIMD tasks.
// Also note that your final grade will be capped at 10.

struct Point
{
	float2 pos;				// current position of the point
	float2 prev_pos;		// position of the point in the previous frame
	float2 fix;				// stationary position; used for the top line of points
	bool fixed;				// true if this is a point in the top line of the cloth
	float restlength[4];	// initial distance to neighbours
};

// Current point positions
static union { float pos_x[GRIDSIZE * GRIDSIZE]; __m128 pos_x4[GRIDSIZE * GRIDSIZE / 4]; };
static union { float pos_y[GRIDSIZE * GRIDSIZE]; __m128 pos_y4[GRIDSIZE * GRIDSIZE / 4]; };
// Previous point positions
static union { float prev_pos_x[GRIDSIZE * GRIDSIZE]; __m128 prev_pos_x4[GRIDSIZE * GRIDSIZE / 4]; };
static union { float prev_pos_y[GRIDSIZE * GRIDSIZE]; __m128 prev_pos_y4[GRIDSIZE * GRIDSIZE / 4]; };
// Stationary positions
static union { float fix_x[GRIDSIZE * GRIDSIZE]; __m128 fix_x4[GRIDSIZE * GRIDSIZE / 4]; };
static union { float fix_y[GRIDSIZE * GRIDSIZE]; __m128 fix_y4[GRIDSIZE * GRIDSIZE / 4]; };
// True for points in the top line of the cloth
static bool is_fixed[GRIDSIZE * GRIDSIZE];
// Initial distances to neighbours
static union { float rest[GRIDSIZE * GRIDSIZE * 4]; __m128 rest4[GRIDSIZE * GRIDSIZE]; };
// Mapping from x, y standard point indices to SoA indices
static int mapping[GRIDSIZE * GRIDSIZE];

// grid access convenience
Point* pointGrid = new Point[GRIDSIZE * GRIDSIZE];
Point& grid( const uint x, const uint y ) { return pointGrid[x + y * GRIDSIZE]; }

// grid offsets for the neighbours via the four links
int xoffset[4] = { 1, -1, 0, 0 }, yoffset[4] = { 0, 0, 1, -1 };

// initialization
void Game::Init() {
	// create the cloth
	for (int y = 0; y < GRIDSIZE; y++) for (int x = 0; x < GRIDSIZE; x++) {
		grid( x, y ).pos.x = 10 + (float)x * ((SCRWIDTH - 100) / GRIDSIZE) + y * 0.9f + Rand( 2 );
		grid( x, y ).pos.y = 10 + (float)y * ((SCRHEIGHT - 180) / GRIDSIZE) + Rand( 2 );
		grid( x, y ).prev_pos = grid( x, y ).pos; // all points start stationary
		if (y == 0) {
			grid( x, y ).fixed = true;
			grid( x, y ).fix = grid( x, y ).pos;
		} else {
			grid( x, y ).fixed = false;
		}
	}
	for (int y = 1; y < GRIDSIZE - 1; y++) for (int x = 1; x < GRIDSIZE - 1; x++) {
		// calculate and store distance to four neighbours, allow 15% slack
		for (int c = 0; c < 4; c++) {
			grid( x, y ).restlength[c] = length( grid( x, y ).pos - grid( x + xoffset[c], y + yoffset[c] ).pos ) * 1.15f;
		}
	}

	// Conversion to SoA
	for ( int y = 0; y < GRIDSIZE; y++ ) {
		// divide into groups of four, each of size GRIDSIZE / 4
		// data layout in quadfloats e.g. for idx 2: **[*]*...****|**[*]*...****|**[*]*...****|**[*]*...****
		for ( int x = 0; x < GRIDSIZE / 4; x++ ) {
			for ( int g = 0; g < 4; g++ ) {
				int base_idx = x * 4 + y * GRIDSIZE;
				int soa_idx  = base_idx + g;

				int x_point = x + g * ( GRIDSIZE / 4 );
				Point& point = grid( x_point, y );
				mapping[x_point + y * GRIDSIZE] = soa_idx;

				pos_x[soa_idx] = point.pos.x;
				pos_y[soa_idx] = point.pos.y;

				prev_pos_x[soa_idx] = point.prev_pos.x;
				prev_pos_y[soa_idx] = point.prev_pos.y;

				fix_x[soa_idx] = point.fix.x;
				fix_y[soa_idx] = point.fix.y;

				is_fixed[soa_idx] = point.fixed;

				rest[( base_idx + 0 ) * 4 + g] = point.restlength[0];
				rest[( base_idx + 1 ) * 4 + g] = point.restlength[1];
				rest[( base_idx + 2 ) * 4 + g] = point.restlength[2];
				rest[( base_idx + 3 ) * 4 + g] = point.restlength[3];
			}
		}
	}
}

// cloth rendering
// NOTE: For this assignment, please do not attempt to render directly on
// the GPU. Instead, if you use GPGPU, retrieve simulation results each frame
// and render using the function below. Do not modify / optimize it.
void Game::DrawGrid() {
	// draw the grid
	screen->Clear( 0 );
	for ( int y = 0; y < ( GRIDSIZE - 1 ); y++ ) {
		for ( int x = 1; x < ( GRIDSIZE - 2 ); x++ ) {
			int idx1 = mapping[x + y * GRIDSIZE];
			int idx2 = mapping[( x + 1 ) + y * GRIDSIZE];
			int idx3 = mapping[x + ( y + 1 ) * GRIDSIZE];
			screen->Line( pos_x[idx1], pos_y[idx1], pos_x[idx2], pos_y[idx2], 0xffffff );
			screen->Line( pos_x[idx1], pos_y[idx1], pos_x[idx3], pos_y[idx3], 0xffffff );
		}
	}

	for ( int y = 0; y < ( GRIDSIZE - 1 ); y++ ) {
		int idx1 = mapping[( GRIDSIZE - 2 ) + y * GRIDSIZE];
		int idx2 = mapping[( GRIDSIZE - 2 ) + ( y + 1 ) * GRIDSIZE];
		screen->Line( pos_x[idx1], pos_y[idx1], pos_x[idx2], pos_y[idx2], 0xffffff );
	}
}

// cloth simulation
// This function implements Verlet integration (see notes at top of file).
// Important: when constraints are applied, typically two points are
// drawn together to restore the rest length. When running on the GPU or
// when using SIMD, this will only work if the two vertices are not
// operated upon simultaneously (in a vector register, or in a warp).
float magic = 0.11f;
__m128 gravity4 = _mm_set1_ps( 0.00000003f );
__m128 magic_chance4 = _mm_set1_ps( 0.03f );
__m128 one4 = _mm_set1_ps( 1 );
__m128 half4 = _mm_set1_ps( 0.5f );
void resolveSprings( int pidx, int nidx, int sidx ) {
	//float2 pointpos = grid( x, y ).pos;
	__m128 px4 = pos_x4[pidx];
	__m128 py4 = pos_y4[pidx];
	//Point& neighbour = grid( x + xoffset[linknr], y + yoffset[linknr] );
	__m128 nx4 = pos_x4[nidx];
	__m128 ny4 = pos_y4[nidx];
	//float2 dir = neighbour.pos - pointpos;
	__m128 dx4 = _mm_sub_ps( nx4, px4 );
	__m128 dy4 = _mm_sub_ps( ny4, py4 );
	//float distance = length( neighbour.pos - pointpos );
	__m128 dist4 = _mm_sqrt_ps( _mm_add_ps( _mm_mul_ps( dx4, dx4 ), _mm_mul_ps( dy4, dy4 ) ) );

	/*float result[4];
	_mm_store_ps( result, dist4 );
	printf( "dist4: %f\t\t%f\t\t%f\t\t%f\n", result[0], result[1], result[2], result[3] );
	_mm_store_ps( result, rest4[sidx] );
	printf( "rest4: %f\t\t%f\t\t%f\t\t%f\n", result[0], result[1], result[2], result[3] );*/

	// Create a mask using the restlength, rest4[ridx] has restlenth[link] for each of the four neighbours of the point
	__m128 mask = _mm_andnot_ps( _mm_cmpord_ps( dist4, dist4 ), _mm_cmpgt_ps( dist4, rest4[sidx] ) );

	/*_mm_store_ps( result, _mm_cmpgt_ps( dist4, rest4[sidx] ) );
	printf( "dmask: %f\t\t%f\t\t%f\t\t%f\n", result[0], result[1], result[2], result[3] );
	_mm_store_ps( result, mask );
	printf( "fmask: %f\t\t%f\t\t%f\t\t%f\n\n", result[0], result[1], result[2], result[3] );*/

	dist4 = _mm_blendv_ps( _mm_setzero_ps(), dist4, mask );
	dx4 = _mm_blendv_ps( _mm_setzero_ps(), dx4, mask );
	dy4 = _mm_blendv_ps( _mm_setzero_ps(), dy4, mask );
	__m128 ones4 = _mm_blendv_ps( _mm_setzero_ps(), one4, mask );
	//float extra = distance / ( grid( x, y ).restlength[linknr] ) - 1;
	__m128 extra4 = _mm_sub_ps( _mm_div_ps( dist4, rest4[sidx] ), ones4 );
	// save some multiplications by multiplying the extra already with 0.5f instead of once for each point and neighbour
	extra4 = _mm_mul_ps( extra4, half4 );
	//pointpos += extra * dir * 0.5f;
	__m128 edx4 = _mm_mul_ps( extra4, dx4 );
	__m128 edy4 = _mm_mul_ps( extra4, dy4 );

	//neighbour.pos -= extra * dir * 0.5f;
	pos_x4[nidx] = _mm_sub_ps( nx4, edx4 );
	pos_y4[nidx] = _mm_sub_ps( ny4, edy4 );
	//grid( x, 0 ).pos = pointpos;
	pos_x4[pidx] = _mm_add_ps( px4, edx4 );
	pos_y4[pidx] = _mm_add_ps( py4, edy4 );
}

void Game::Simulation() {
	// simulation is exected three times per frame; do not change this.
	for( int steps = 0; steps < 3; steps++ ) {
		// verlet integration; apply gravity
		for ( int y = 0; y < GRIDSIZE; y++ ) {
			for ( int x = 0; x < GRIDSIZE / 4; x++ ) {
				int idx = x + y * GRIDSIZE / 4;
				//float2 curpos = grid( x, y ).pos;
				__m128 curr_x4 = pos_x4[idx];
				__m128 curr_y4 = pos_y4[idx];
				//float2 prevpos = grid( x, y ).prev_pos;
				__m128 prev_x4 = prev_pos_x4[idx];
				__m128 prev_y4 = prev_pos_y4[idx];
				//grid( x, y ).prev_pos = curpos;
				prev_pos_x4[idx] = curr_x4;
				prev_pos_y4[idx] = curr_y4;
				//grid( x, y ).pos = curpos + ( curpos - prevpos ) + float2( 0, 0.003f ); // gravity
				curr_x4 = _mm_add_ps( curr_x4, _mm_sub_ps( curr_x4, prev_x4 ) );
				curr_y4 = _mm_add_ps( _mm_add_ps( curr_y4, _mm_sub_ps( curr_y4, prev_y4 ) ), gravity4 );
				//if ( Rand( 10 ) < 0.03f ) grid( x, y ).pos += float2( Rand( 0.02f + magic ), Rand( 0.12f ) );
				// avoid conditional code by using a mask
				__m128 rand_d = _mm_set_ps( Rand( 10 ), Rand( 10 ), Rand( 10 ), Rand( 10 ) );
				__m128 mask = _mm_cmplt_ps( rand_d, magic_chance4 );
				// use the mask to extract what we want to apply from rand_x and rand_y
				float range = 0.02f + magic;
				__m128 rand_x = _mm_and_ps( mask, _mm_set_ps( Rand( range ), Rand( range ), Rand( range ), Rand( range ) ) );
				__m128 rand_y = _mm_and_ps( mask, _mm_set_ps( Rand( 0.12f ), Rand( 0.12f ), Rand( 0.12f ), Rand( 0.12f ) ) );
				// do the addition
				pos_x4[idx] = _mm_add_ps( curr_x4, rand_x );
				pos_y4[idx] = _mm_add_ps( curr_y4, rand_y );
			}
		}

		// slowly increases the chance of anomalies
		magic += 0.0002f;
	}
}

void Game::Tick( float a_DT ) {
	// update the simulation
	Timer tm;
	tm.reset();
	Simulation();
	float elapsed1 = tm.elapsed();

	// draw the grid
	tm.reset();
	DrawGrid();
	float elapsed2 = tm.elapsed();

	// display statistics
	char t[128];
	sprintf( t, "ye olde ruggeth cloth simulation: %5.1f ms", elapsed1 * 1000 );
	screen->Print( t, 2, SCRHEIGHT - 24, 0xffffff );
	sprintf( t, "                       rendering: %5.1f ms", elapsed2 * 1000 );
	screen->Print( t, 2, SCRHEIGHT - 14, 0xffffff );
}