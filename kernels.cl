#include "cl/tools.cl"

__kernel void apply_gravity( int grid_size, float magic, __global float2* positions, __global float2* previous_positions, __global uint* seeds ) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int idx = x + y * grid_size;

	float2 curpos = positions[idx];
	float2 prevpos = previous_positions[idx];
	positions[idx] += (curpos - prevpos) + (float2)( 0, 0.003f ); // apply gravity
	previous_positions[idx] = curpos;

	// get the chance of magic happening
	float magic_chance = RandomFloat( &seeds[idx] ) * 10.0f;
	if ( magic_chance < 0.03f ) {
		float magic_x = RandomFloat( &seeds[idx] ) * ( 0.02f + magic );
		float magic_y = RandomFloat( &seeds[idx] ) * 0.12f;
		positions[idx] += (float2)( magic_x, magic_y );
	}
}