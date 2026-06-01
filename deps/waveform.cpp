#include "waveform.h"

/********* ########### **********/
/********* Oscillators **********/
/********* ########### **********/



/********* ###### **********/
/********* Signal **********/
/********* ###### **********/

const size_t Signal::size() {
    return signal.size();
}

Signal Signal::operator+( const Signal& rhs ) {

}

Signal& Signal::operator+=( const Signal& rhs ) {
    // TODO: Finish this one
    auto leftIt = this->signal.begin();
    auto rightIt = rhs.signal.begin();
    return *this;
}

void Signal::writeWAV(std::string &filename) {
    /* TODO */

}