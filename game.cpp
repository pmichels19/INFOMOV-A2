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

struct Point {
	float2 pos;				// current position of the point
	float2 prev_pos;		// position of the point in the previous frame
	float2 fix;				// stationary position; used for the top line of points
	bool fixed;				// true if this is a point in the top line of the cloth
	float restlength[4];	// initial distance to neighbours
};

static uint seeds[GRIDSIZE * GRIDSIZE];
static float2 positions[GRIDSIZE * GRIDSIZE];
static float2 previous_positions[GRIDSIZE * GRIDSIZE];
static float2 fix[GRIDSIZE * GRIDSIZE];
static float4 restlength[GRIDSIZE * GRIDSIZE];

// grid access convenience
Point* pointGrid = new Point[GRIDSIZE * GRIDSIZE];
Point& grid( const uint x, const uint y ) { return pointGrid[x + y * GRIDSIZE]; }

// grid offsets for the neighbours via the four links
int xoffset[4] = { 1, -1, 0, 0 }, yoffset[4] = { 0, 0, 1, -1 };

// initialization
void Game::Init() {
	// Initialize OpenCL
	if ( !gravity_kernel ) {
		Kernel::InitCL();
		gravity_kernel = new Kernel( "kernels.cl", "apply_gravity" );
		seed_buffer = new Buffer( GRIDSIZE * GRIDSIZE * sizeof( uint ), seeds );
		curr_position_buffer = new Buffer( GRIDSIZE * GRIDSIZE * 2 * sizeof( float ), positions );
		prev_position_buffer = new Buffer( GRIDSIZE * GRIDSIZE * 2 * sizeof( float ), previous_positions );
	}

	// create the cloth
	for ( int y = 0; y < GRIDSIZE; y++ ) {
		for ( int x = 0; x < GRIDSIZE; x++ ) {
			int idx = x + y * GRIDSIZE;
			positions[idx].x = 10 + (float) x * ( ( SCRWIDTH - 100 ) / GRIDSIZE ) + y * 0.9f + Rand( 2 );
			positions[idx].y = 10 + (float) y * ( ( SCRHEIGHT - 180 ) / GRIDSIZE ) + Rand( 2 );
			previous_positions[idx] = positions[idx];
			seeds[idx] = RandomUInt() + 1;

			if ( y == 0 ) {
				fix[idx] = positions[idx];
			}
		}
	}

	for ( int y = 1; y < GRIDSIZE - 1; y++ ) {
		for ( int x = 1; x < GRIDSIZE - 1; x++ ) {
			int pidx = x + y * GRIDSIZE;
			
			for ( int c = 0; c < 4; c++ ) {
				int nidx = ( x + xoffset[c] ) + ( y + yoffset[c] ) * GRIDSIZE;
				restlength[pidx][c] = length( positions[pidx] - positions[nidx]) * 1.15f;
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
	for (int y = 0; y < (GRIDSIZE - 1); y++) for (int x = 1; x < (GRIDSIZE - 2); x++) {
		const float2 p1 = positions[( x + 0 ) + ( y + 0 ) * GRIDSIZE];
		const float2 p2 = positions[( x + 1 ) + ( y + 0 ) * GRIDSIZE];
		const float2 p3 = positions[( x + 0 ) + ( y + 1 ) * GRIDSIZE];
		screen->Line( p1.x, p1.y, p2.x, p2.y, 0xffffff );
		screen->Line( p1.x, p1.y, p3.x, p3.y, 0xffffff );
	}

	for (int y = 0; y < (GRIDSIZE - 1); y++) {
		const float2 p1 = positions[( GRIDSIZE - 2 ) + ( y + 0 ) * GRIDSIZE];
		const float2 p2 = positions[( GRIDSIZE - 2 ) + ( y + 1 ) * GRIDSIZE];
		screen->Line( p1.x, p1.y, p2.x, p2.y, 0xffffff );
	}
}

// cloth simulation
// This function implements Verlet integration (see notes at top of file).
// Important: when constraints are applied, typically two points are
// drawn together to restore the rest length. When running on the GPU or
// when using SIMD, this will only work if the two vertices are not
// operated upon simultaneously (in a vector register, or in a warp).
float magic = 0.11f;
void Game::Simulation() {
	// simulation is exected three times per frame; do not change this.
	for( int steps = 0; steps < 3; steps++ ) {
		// verlet integration; apply gravity
		seed_buffer->CopyToDevice();
		curr_position_buffer->CopyToDevice();
		prev_position_buffer->CopyToDevice();

		gravity_kernel->SetArguments( GRIDSIZE, magic, curr_position_buffer, prev_position_buffer, seed_buffer );
		gravity_kernel->Run( GRIDSIZE * GRIDSIZE );

		seed_buffer->CopyFromDevice();
		curr_position_buffer->CopyFromDevice();
		prev_position_buffer->CopyFromDevice( true );
		//	if (Rand( 10 ) < 0.03f) positions[idx] += float2(Rand(0.02f + magic), Rand(0.12f));

		magic += 0.0002f; // slowly increases the chance of anomalies
		// apply constraints; 4 simulation steps: do not change this number.
		for (int i = 0; i < 4; i++) {
			for (int y = 1; y < GRIDSIZE - 1; y++) for (int x = 1; x < GRIDSIZE - 1; x++) {
				int pidx = x + y * GRIDSIZE;
				float2 pointpos = positions[pidx];
				// use springs to four neighbouring points
				for (int linknr = 0; linknr < 4; linknr++) {
					int nidx = ( x + xoffset[linknr] ) + ( y + yoffset[linknr] ) * GRIDSIZE;
					float distance = length( positions[nidx] - pointpos);
					if (!isfinite( distance )) {
						// warning: this happens; sometimes vertex positions 'explode'.
						continue;
					}

					if (distance > restlength[pidx][linknr]) {
						// pull points together
						float extra = distance / ( restlength[pidx][linknr] ) - 1;
						float2 dir = positions[nidx] - pointpos;
						pointpos += extra * dir * 0.5f;
						positions[nidx] -= extra * dir * 0.5f;
					}
				}

				positions[pidx] = pointpos;
			}
			// fixed line of points is fixed.
			for (int x = 0; x < GRIDSIZE; x++) {
				positions[x] = fix[x];
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