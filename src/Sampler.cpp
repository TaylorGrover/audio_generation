#include <iostream>
#include <qt5/QtCore/QScopedPointer>
#include <qt5/QtWidgets/QWidget>
#include <qt5/QtWidgets/QApplication>
#include "sampler_window.h"

int main(int argc, char *argv[]) {

	QApplication* app = new QApplication(argc, argv);

	SamplerMain mainWindow;
	mainWindow.show();

	return app->exec();
}
