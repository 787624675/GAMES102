#pragma once

#include <UGM/UGM.h>
struct Slope {
	float l;
	float r;
};

struct Ratio {
	float l;
	float r;
	Ratio() { l = r = 1.f; }
};
struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	std::vector<Ubpa::pointf2> ltangent;
	std::vector<Ubpa::pointf2> rtangent;
	std::vector<Ratio> tangent_ratio;
	std::vector<Slope> xk;
	std::vector<Slope> yk;
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
	bool bezier{ false };  // Creat editable cubic spline cruve
	bool edit{ false };   // edit mode
	bool g0{ false };   // edit mode
	bool g1{ false };   // edit mode
	bool edit_tan{ false };
	bool editing_tan{ false };
	bool edit_flag{ false };   // whather the tangent line have been edited
	bool chaiukin_sub{ false };
	bool cubic_sub{ false };
	bool inter_sub{ false };
	int editing_tan_index;
	int chaiukin_num;
	float alpha;

};

#include "details/CanvasData_AutoRefl.inl"
