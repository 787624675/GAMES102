#pragma once

#include <UGM/UGM.h>

struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	bool adding_line{ false };
	bool editing{ false };
	bool showCruve1{ false };  // 1:Polynomial function
	bool showCruve2{ false };  // 2:Guess function
	bool showCruve3{ false };  // 3.Power function
	bool showCruve4{ false };  // 4.Ridge Regression
	bool showCruve5{ false };  // 5:Equidistant (uniform) parameterization
	bool showCruve6{ false };  // 6:Chordal parameterization
	bool showCruve7{ false };  // 7.Centripetal parameterization
	bool showCruve8{ false };  // 8.Foley parameterization
	bool cubic{ false };  // Creat editable cubic spline cruve
	bool edit{ false };   // �༭ģʽ

};

#include "details/CanvasData_AutoRefl.inl"
