#include "sampler_window.h"

SamplerMain::SamplerMain() {
    std::cout << "asdfasdfasdf" << std::endl;
    setupLayout();
}

void SamplerMain::setupLayout() {
    // Set Window position and dimensions
    QScreen* screen = QGuiApplication::primaryScreen();
    screenwidth = screen->size().width();
    screenheight = screen->size().height();

    setMinimumWidth(screenwidth * 2 / 3);
    setMinimumHeight(2 * screenheight / 3);
}