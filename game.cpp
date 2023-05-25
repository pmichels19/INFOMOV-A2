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
		for ( int x = 0; x < GRIDSIZE; x++ ) {
			int idx = x + y * GRIDSIZE;

			pos_x[idx] = grid( x, y ).pos.x;
			pos_y[idx] = grid( x, y ).pos.y;

			prev_pos_x[idx] = grid( x, y ).prev_pos.x;
			prev_pos_y[idx] = grid( x, y ).prev_pos.y;

			fix_x[idx] = grid( x, y ).fix.x;
			fix_y[idx] = grid( x, y ).fix.y;

			is_fixed[idx] = grid( x, y ).fixed;

			rest[idx + 0] = grid( x, y ).restlength[0];
			rest[idx + 1] = grid( x, y ).restlength[1];
			rest[idx + 2] = grid( x, y ).restlength[2];
			rest[idx + 3] = grid( x, y ).restlength[3];
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
	for ( int y = 0; y < ( GRIDSIZE - 1 ); y++ ) for ( int x = 1; x < ( GRIDSIZE - 2 ); x++ ) {
		int idx1 = x + y * GRIDSIZE;
		int idx2 = ( x + 1 ) + y * GRIDSIZE;
		int idx3 = x + ( y + 1 ) * GRIDSIZE;
		screen->Line( pos_x[idx1], pos_y[idx1], pos_x[idx2], pos_y[idx2], 0xFFFFFF );
		screen->Line( pos_x[idx1], pos_y[idx1], pos_x[idx3], pos_y[idx3], 0xFFFFFF );
	}
	for ( int y = 0; y < ( GRIDSIZE - 1 ); y++ ) {
		int idx1 = ( GRIDSIZE - 2 ) + y * GRIDSIZE;
		int idx2 = ( GRIDSIZE - 2 ) + ( y + 1 ) * GRIDSIZE;
		screen->Line( pos_x[idx1], pos_y[idx1], pos_x[idx2], pos_y[idx2], 0xFFFFFF );
	}

	// draw the grid
	//screen->Clear( 0 );
	//for (int y = 0; y < (GRIDSIZE - 1); y++) {
	//	for (int x = 1; x < (GRIDSIZE - 2); x++) {
	//		int idx1 = x + y * GRIDSIZE;
	//		int idx2 = ( x + 1 ) + y * GRIDSIZE;
	//		int idx3 = x + ( y + 1 ) * GRIDSIZE;
	//		screen->Line( pos_x[idx1], pos_y[idx1], pos_x[idx2], pos_y[idx2], 0xffffff );
	//		screen->Line( pos_x[idx1], pos_y[idx1], pos_x[idx3], pos_y[idx3], 0xffffff );
	//	}
	//}

	//for (int y = 0; y < (GRIDSIZE - 1); y++) {
	//	int idx1 = ( GRIDSIZE - 2 ) + y * GRIDSIZE;
	//	int idx2 = ( GRIDSIZE - 2 ) + ( y + 1 ) * GRIDSIZE;
	//	screen->Line( pos_x[idx1], pos_y[idx1], pos_x[idx2], pos_y[idx2], 0xffffff );
	//}
}

// cloth simulation
// This function implements Verlet integration (see notes at top of file).
// Important: when constraints are applied, typically two points are
// drawn together to restore the rest length. When running on the GPU or
// when using SIMD, this will only work if the two vertices are not
// operated upon simultaneously (in a vector register, or in a warp).
float magic = 0.11f;
__m128 gravity4 = _mm_set1_ps( 0.003f );
__m128 magic_chance4 = _mm_set1_ps( 0.03f );
__m128 one4 = _mm_set1_ps( 1 );
__m128 half4 = _mm_set1_ps( 0.5f );
void Game::Simulation() {
	// simulation is exected three times per frame; do not change this.
	for( int steps = 0; steps < 3; steps++ ) {
		// TODO: task 1 - SIMD-ify this for-loop
		// verlet integration; apply gravity
		for (int y = 0; y < GRIDSIZE / 4; y++) {
			for (int x = 0; x < GRIDSIZE; x++) {
				int idx = x + y * GRIDSIZE;
				//float2 curpos = grid( x, y ).pos;
				__m128 curr_x4 = pos_x4[idx];
				__m128 curr_y4 = pos_y4[idx];
				//float2 prevpos = grid( x, y ).prev_pos;
				__m128 prev_x4 = prev_pos_x4[idx];
				__m128 prev_y4 = prev_pos_y4[idx];
				//grid( x, y ).pos = curpos + ( curpos - prevpos ) + float2( 0, 0.003f ); // gravity
				pos_x4[idx] = _mm_add_ps( curr_x4, _mm_sub_ps( curr_x4, prev_x4 ) );
				pos_y4[idx] = _mm_add_ps( _mm_add_ps( curr_y4, _mm_sub_ps( curr_y4, prev_y4 ) ), gravity4 );
				//grid( x, y ).prev_pos = curpos;
				prev_pos_x4[idx] = curr_x4;
				prev_pos_y4[idx] = curr_y4;
				//if ( Rand( 10 ) < 0.03f ) grid( x, y ).pos += float2( Rand( 0.02f + magic ), Rand( 0.12f ) );
				// avoid conditional code by using a mask
				__m128 rand_d = _mm_set_ps( Rand( 10 ), Rand( 10 ), Rand( 10 ), Rand( 10 ) );
				__m128 mask = _mm_cmplt_ps( rand_d, magic_chance4 );
				// use the mask to extract what we want to apply from rand_x and rand_y
				float range = 0.02f + magic;
				__m128 rand_x = _mm_and_ps( mask, _mm_set_ps( Rand( range ), Rand( range ), Rand( range ), Rand( range ) ) );
				__m128 rand_y = _mm_and_ps( mask, _mm_set_ps( Rand( 0.12f ), Rand( 0.12f ), Rand( 0.12f ), Rand( 0.12f ) ) );
				// do the addition
				pos_x4[idx] = _mm_add_ps( pos_x4[idx], rand_x );
				pos_y4[idx] = _mm_add_ps( pos_y4[idx], rand_y );
			}
		}
		// verlet integration; apply gravity
		for ( int y = 0; y < GRIDSIZE; y++ ) for ( int x = 0; x < GRIDSIZE; x++ ) {
			float2 curpos = grid( x, y ).pos, prevpos = grid( x, y ).prev_pos;
			grid( x, y ).pos += ( curpos - prevpos ) + float2( 0, 0.003f ); // gravity
			grid( x, y ).prev_pos = curpos;
			if ( Rand( 10 ) < 0.03f ) grid( x, y ).pos += float2( Rand( 0.02f + magic ), Rand( 0.12f ) );
		}

		// slowly increases the chance of anomalies
		magic += 0.0002f;

		for ( int y = 0; y < GRIDSIZE; y++ ) {
			for ( int x = 0; x < GRIDSIZE; x++ ) {
				int idx = x + y * GRIDSIZE;
				float2 pointPos( pos_x[idx], pos_y[idx] );
				grid( x, y ).pos = pointPos;
				float2 prevPos( prev_pos_x[idx], prev_pos_y[idx] );
				grid( x, y ).prev_pos = prevPos;
			}
		}

		// apply constraints; 4 simulation steps: do not change this number.
		for ( int i = 0; i < 4; i++ ) {
			for ( int y = 1; y < GRIDSIZE - 1; y++ ) {
				for ( int x = 1; x < GRIDSIZE - 1; x++ ) {
					float2 pointpos = grid( x, y ).pos;
					// use springs to four neighbouring points
					for ( int linknr = 0; linknr < 4; linknr++ ) {
						Point& neighbour = grid( x + xoffset[linknr], y + yoffset[linknr] );
						float distance = length( neighbour.pos - pointpos );
						if ( !isfinite( distance ) ) {
							// warning: this happens; sometimes vertex positions 'explode'.
							continue;
						}

						if ( distance > grid( x, y ).restlength[linknr] ) {
							// pull points together
							float extra = distance / ( grid( x, y ).restlength[linknr] ) - 1;
							float2 dir = neighbour.pos - pointpos;
							pointpos += extra * dir * 0.5f;
							neighbour.pos -= extra * dir * 0.5f;
						}
					}

					grid( x, y ).pos = pointpos;
				}
			}

			// fixed line of points is fixed.
			for ( int x = 0; x < GRIDSIZE; x++ ) {
				grid( x, 0 ).pos = grid( x, 0 ).fix;
			}
		}

		// temporary fix, make sure the 4 simulation steps are saved properly to the SoA
		for ( int y = 0; y < GRIDSIZE; y++ ) {
			for ( int x = 0; x < GRIDSIZE; x++ ) {
				int idx = x + y * GRIDSIZE;
				pos_x[idx] = grid( x, y ).pos.x;
				pos_y[idx] = grid( x, y ).pos.y;
				prev_pos_x[idx] = grid( x, y ).prev_pos.x;
				prev_pos_y[idx] = grid( x, y ).prev_pos.y;
			}
		}
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