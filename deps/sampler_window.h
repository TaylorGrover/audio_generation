#ifndef SAMPLER_WINDOW_H
#define SAMPLER_WINDOW_H

#include <iostream>

#include <qt5/QtCore/QObject>
#include <qt5/QtCore/QPoint>
#include <qt5/QtCore/QSettings>
#include <qt5/QtCore/QSize>
#include <qt5/QtCore/QString>
#include <qt5/QtGui/QGuiApplication>
#include <qt5/QtGui/QScreen>

#include <qt5/QtWidgets/QCheckBox>
#include <qt5/QtWidgets/QLabel>
#include <qt5/QtWidgets/QPushButton>
#include <qt5/QtWidgets/QFormLayout>
#include <qt5/QtWidgets/QGridLayout>
#include <qt5/QtWidgets/QMainWindow>
#include <qt5/QtWidgets/QStyle>

class SamplerMain : public QMainWindow {
    Q_OBJECT

private:
    int screenwidth, screenheight;
    void setup();
    void createTopLevelWidgets();

    QGridLayout* mainGrid;
    QWidget* mainWidget;
    QPushButton* testButton;

public:
    SamplerMain();


};

#endif // SAMPLER_WINDOW_H