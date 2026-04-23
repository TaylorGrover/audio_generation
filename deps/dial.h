#ifndef DIAL_H
#define DIAL_H

#include <qt5/QtWidgets/QWidget>

class Dial : public QWidget {
    Q_OBJECT

private:
    void setupDial();

public: 
    Dial();
    Dial(QWidget* parent);
};

#endif 