#ifndef SAMPLER_WINDOW_H
#define SAMPLER_WINDOW_H

#include <iostream>

#include <qt5/QtCore/QObject>
#include <qt5/QtCore/QSettings>
#include <qt5/QtCore/QSize>
#include <qt5/QtGui/QGuiApplication>
#include <qt5/QtGui/QScreen>
#include <qt5/QtWidgets/QMainWindow>

class SamplerMain : public QMainWindow {
    Q_OBJECT

private:
    int screenwidth, screenheight;

public:
    SamplerMain();

    void setupLayout();

};

#endif // SAMPLER_WINDOW_H