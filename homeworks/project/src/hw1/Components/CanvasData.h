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
	bool hermit{ false };
	bool edit{ false };   // edit mode
	int edit_index = 1;


	int fitting_type{ 0 };
	bool enable_add_point{ true };
	bool adding_last_point{ false };
	int edit_point{ 0 };
	int editing_index = -1;
	bool enable_move_point{ false };
	int editing_tan_index = 0; // < 0 is left, > 0 is right, real index = this - 1
	bool enable_move_tan{ false };
	void pop_back() {
		points.pop_back();
		ltangent.pop_back();
		rtangent.pop_back();
		xk.pop_back();
		yk.pop_back();
		tangent_ratio.pop_back();
	}
	void clear() {
		points.clear();
		ltangent.clear();
		rtangent.clear();
		xk.clear();
		yk.clear();
		tangent_ratio.clear();
	}
	void push_back(const Ubpa::pointf2& p) {
		points.push_back(p);
		ltangent.push_back(Ubpa::pointf2());
		rtangent.push_back(Ubpa::pointf2());
		xk.push_back(Slope());
		yk.push_back(Slope());
		tangent_ratio.push_back(Ratio());
	}


};

#include "details/CanvasData_AutoRefl.inl"
