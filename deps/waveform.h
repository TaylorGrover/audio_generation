#include <algorithm>
#include <cmath>
#include <iostream>
#include <ranges>
#include <utility>
#include <vector>

class Signal {
private:
    std::vector<float> amplitudes;

public:
    Signal(std::vector<float>& amplitudes) : amplitudes(amplitudes) {}; 
    Signal();

    Signal operator+ ( const Signal& rhs );
    Signal& operator+=( const Signal& rhs );

    const size_t size(); 

    const Signal& getAmplitudes();

    void writeWAV(std::string &filename);
};

class UnitGenerator {
private:
    Signal out;

public:
    Signal generate(); 
};

class Oscillator : UnitGenerator {

private:
    Signal oscillator;

public:
    virtual Signal generateOscillation(float duration) = 0;

};

class Triangle : Oscillator {
public:
    Signal generateOscillation(float duration);
};


class Patch {
private:
    std::vector<UnitGenerator> ugs;

public:
    virtual Signal generateSignal(float duration) = 0;
};