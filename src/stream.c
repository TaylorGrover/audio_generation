#include <math.h>
#include <stdio.h>
#include <string.h>
#include <pulse/simple.h>
#include <pulse/error.h>

#define BUFSIZE 1024

typedef struct WAVE_T {
	
} wave_t;

void free_wave(wave_t* wave);
void gen_wave(wave_t* wave, int duration, float frequency, int sr);


int main(int argc, char* argv[])
{
	pa_simple *s = NULL;
	pa_sample_spec ss;
	ss.format = PA_SAMPLE_S16LE;
	ss.rate = 44100;
	ss.channels = 2;
	int error;

	s = pa_simple_new(NULL, "TossUp", PA_STREAM_PLAYBACK, NULL, "Playback", &ss, NULL, NULL, &error);

	unsigned char buf[BUFSIZE];
	// Contains audio data
	
	for(int i = 0; i < ss.rate * 1; i++) {
		pa_simple_write(s, buf, sizeof(buf), &error);
	}

	pa_simple_free(s);
	return 0;
}

void free_wave(wave_t* wave)
{
	
}


