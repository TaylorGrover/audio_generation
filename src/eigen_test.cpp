#include <cmath>
#include <eigen3/Eigen/Eigen>
#include <iostream>

#define MAX_ARRAYS 15

using Eigen::ArrayXd;


ArrayXd sine_bend_centered(float duration, float freq, float bend_dist, float osc_freq, int sample_rate)
{
	int n_samples = (int) (duration * sample_rate);
	ArrayXd t = ArrayXd::LinSpaced(n_samples, 0, duration);

	float freq_carrier = 2 * M_PI * freq;
	float freq_message = 2 * M_PI * osc_freq;
	float p = 2 * M_PI * freq * (pow(2, bend_dist / 12.0) - 1);

	t = (freq_carrier * t - p / freq_message * (freq_message * t).cos()).sin();
	return t;
}

int main()
{
	ArrayXd bends[MAX_ARRAYS];
	for(int i = 0; i < MAX_ARRAYS; i++) {
		ArrayXd bend = sine_bend_centered(10, 43.65, .5, .2, 44100);
		//std::cout << bend[0] << std::endl;
		bends[i] = bend;
	}
	return 0;
}
