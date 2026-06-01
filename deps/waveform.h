#include <algorithm>
#include <cmath>
#include <iostream>
#include <ranges>
#include <utility>
#include <vector>

class Signal {
private:
    std::vector<float> signal;

public:
    Signal(std::vector<float>& sig) : signal(sig) {}; 

    Signal operator+ ( const Signal& rhs );
    Signal& operator+=( const Signal& rhs );

    const size_t size(); 

    const Signal& getSignal();

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