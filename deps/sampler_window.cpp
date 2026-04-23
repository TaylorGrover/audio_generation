#include "sampler_window.h"

SamplerMain::SamplerMain() 
{
    QScreen* screen = QGuiApplication::primaryScreen();
    screenwidth = screen->size().width();
    screenheight = screen->size().height();
    setup();
}

void SamplerMain::setup() 
{
    // Set Window dimensions and position
    setMinimumWidth(screenwidth * 2 / 3);
    setMinimumHeight(screenheight * 2 / 3);
    move(screenwidth/6, screenheight/6);

    // Set window style
    QString styleString(
        "QWidget {"\
        "   background: #888888;"\
        "}"\
        "QPushButton {"\
        "   color: #ffffff;"\
        "   background-color: #888888;"\
        "}"
    );
    setStyleSheet(styleString);

    createTopLevelWidgets();
}

/**
 * Outermost widgets on the main window
*/
void SamplerMain::createTopLevelWidgets()
{
    mainWidget = new QWidget;
    mainGrid = new QGridLayout;
    this->setCentralWidget(mainWidget);
    
    testButton = new QPushButton("Test");
    mainGrid->addWidget(testButton, 0, 0);

    mainWidget->setLayout(mainGrid);
}