#include "waveform.h"

/********* ########### **********/
/********* Oscillators **********/
/********* ########### **********/



/********* ###### **********/
/********* Signal **********/
/********* ###### **********/

const size_t Signal::size() {
    return amplitudes.size();
}

Signal Signal::operator+( const Signal& rhs ) {
    std::vector<float> amps(std::max(this->size(), rhs.amplitudes.size()));
    int index = 0;
    auto leftIt = this->amplitudes.begin();
    auto rightIt = rhs.amplitudes.begin();
    for(; 
        leftIt != this->amplitudes.end() 
            && rightIt != rhs.amplitudes.end(); 
        leftIt++, rightIt++, index++
    ) {
        amps[index] = *leftIt + *rightIt;
    }
    while( leftIt != this->amplitudes.begin() ) {
        amps[index++] = *leftIt;
        leftIt++;
    }
    while ( rightIt != rhs.amplitudes.end() ) {
        amps[index++] = *rightIt;
        rightIt++;
    }
    Signal newSignal(amps);
    return newSignal;
}

Signal& Signal::operator+=( const Signal& rhs ) {
    // TODO: Finish this one
    auto leftIt = this->amplitudes.begin();
    auto rightIt = rhs.amplitudes.begin();
    return *this;
}

void Signal::writeWAV(std::string &filename) {
    /* TODO */

}