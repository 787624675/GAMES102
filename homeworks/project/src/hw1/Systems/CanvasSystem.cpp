#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include <Eigen/Core>

#include <Eigen/QR>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace Ubpa;
constexpr int sample_num = 50;
constexpr float base_tangent_len = 50.f;
constexpr float t_step = 1.f / (sample_num - 1);
constexpr float point_radius = 3.f;
constexpr ImU32 line_col = IM_COL32(39, 117, 182, 255);
constexpr ImU32 edit_line_col = IM_COL32(39, 117, 182, 100);
constexpr ImU32 normal_point_col = IM_COL32(255, 0, 0, 255);
constexpr ImU32 select_point_col = IM_COL32(255, 0, 0, 100);
constexpr ImU32 slope_col = IM_COL32(122, 115, 116, 255);
constexpr ImU32 select_slope_col = IM_COL32(122, 115, 116, 100);
double S_1(Eigen::VectorXd m, Eigen::VectorXd h, Eigen::VectorXd myvar, Eigen::VectorXd T_1, int segment1, double t1) {
	double res = m[segment1] * pow(T_1[segment1 + 1] - t1, 3) / (6 * h[segment1]) 
		+ m[segment1 + 1] * pow(t1 - T_1[segment1], 3) / (6 * h[segment1]) 
		+ (myvar[segment1 + 1] / h[segment1] - m[segment1 + 1] * h[segment1] / 6) * (t1 - T_1[segment1])
		+ (myvar[segment1] / h[segment1] - m[segment1] * h[segment1] / 6) * (T_1[segment1 + 1] - t1);
	return res;
}
double find_min_x(Eigen::VectorXd xvals) {
	int res = xvals[0];
	for (int i = 1;i < xvals.size();i++) {
		if (xvals[i] < res) res = xvals[i];
	}
	return res;
}
double find_max_x(Eigen::VectorXd xvals) {
	int res = xvals[0];
	for (int i = 1;i < xvals.size();i++) {
		if (xvals[i] > res) res = xvals[i];
	}
	return res;
}
double polyeval(Eigen::VectorXd coeffs, double x){
	double result = 0.0;
	for (int i = 0; i < coeffs.size(); i++)
	{
		result += coeffs[i] * pow(x, i);
	}
	return result;
}
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order)
{
	assert(xvals.size() == yvals.size());
	assert(order >= 1 && order <= xvals.size() - 1);
	Eigen::MatrixXd A(xvals.size(), order + 1);

	for (int i = 0; i < xvals.size(); i++)
		A(i, 0) = 1.0;

	for (int j = 0; j < xvals.size(); j++)
	{
		for (int i = 0; i < order; i++)
			A(j, i + 1) = A(j, i) * xvals(j);
	}
	auto Q = A.householderQr();
	auto result = Q.solve(yvals);
	return result;
}
double gausseval(Eigen::VectorXd coeffs, Eigen::VectorXd xvals, double x, double delta = 1) {
	double result = coeffs[0];  // b0
	for (int i = 1; i < coeffs.size(); i++)
	{
		result += coeffs[i] * exp(-pow(x - xvals[i], 2) / (2 * delta * delta));
	}
	return result;
}
Eigen::VectorXd gaussfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, double delta  = 1.0)
{
	assert(xvals.size() == yvals.size());
	Eigen::MatrixXd A(xvals.size(), xvals.size());
	for (int i = 0; i < xvals.size(); i++) {
		for (int j = 0; j < xvals.size(); j++) {
			if (j == 0) {
				A(i, j) = 1;
			}
			else {
				A(i, j) = exp(-pow(xvals[i] - xvals[j], 2) / (2 * delta * delta));
			}
		}
	}
	auto Q = A.householderQr();
	auto result = Q.solve(yvals);
	return result;
}
// Poly  approximation function
Eigen::VectorXd polyappro(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order)
{
	assert(xvals.size() == yvals.size());
	Eigen::MatrixXd A(xvals.size(), order + 1);
	for (int i = 0; i < xvals.size();i++) {
		for (int j = 0; j < order + 1;j++) {
			A(i, j) = pow(xvals[i], j);
		}
	}
	MatrixXd B = A.transpose() * A;
	auto result = B.inverse() * A.transpose() * yvals;
	return result;
}
Eigen::VectorXd polyappro_op(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order, double lambda=1)
{
	assert(xvals.size() == yvals.size());
	Eigen::MatrixXd A( xvals.size(),order + 1 );
	
	for (int i = 0; i < xvals.size();i++) {
		for (int j = 0; j < order+1;j++) {
			A(i, j) = pow(xvals[i], j);
		}
	}
	MatrixXd B = A.transpose() * A;
	for (int i = 0; i < order+1;i++) {
		for (int j = 0; j < order+1;j++) {
			if (i == j) {
				B(i, j) += lambda;
			}
			
		}
	}
	auto result = B.inverse()*A.transpose()*yvals;
	return result;
}
inline float h0(const float x0, const float x1, const float x) {
	return (1.f + 2.f * (x - x0) / (x1 - x0)) * ((x - x1) * (x - x1)) / ((x0 - x1) * (x0 - x1));
}
inline float h1(const float x0, const float x1, const float x) {
	return h0(x1, x0, x);
}
inline float H0(const float x0, const float x1, const float x) {
	return (x - x0) * ((x - x1) * (x - x1)) / ((x0 - x1) * (x0 - x1));
}
inline float H1(const float x0, const float x1, const float x) {
	return H0(x1, x0, x);
}
constexpr int Bezier_num = 100;
void DeCasteljau(std::vector<Ubpa::pointf2>& ret, Ubpa::pointf2* p) {
	const float step = 1.f / (Bezier_num - 1);
	for (float t = 0.f; t <= 1.f; t += step) {
		float x = std::pow((1 - t), 3) * p[0][0] + 3 * t * (1 - t) * (1 - t) * p[1][0] + 3 * t * t * (1 - t) * p[2][0] + t * t * t * p[3][0];
		float y = std::pow((1 - t), 3) * p[0][1] + 3 * t * (1 - t) * (1 - t) * p[1][1] + 3 * t * t * (1 - t) * p[2][1] + t * t * t * p[3][1];
		ret.push_back(Ubpa::pointf2(x, y));
	}
}

void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;

		if (ImGui::Begin("Canvas")) {
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);
			ImGui::Text("Mouse Left: drag to add lines,\nMouse Right: drag to scroll, click for context menu.");
			ImGui::Checkbox("polynomial function", &data->showCruve1);
			ImGui::Checkbox("Gauss function", &data->showCruve2);
			ImGui::Checkbox("Power function", &data->showCruve3);
			ImGui::Checkbox("Ridge Regression", &data->showCruve4);
			ImGui::Checkbox("Equidistant (uniform) parameterization", &data->showCruve5);
			ImGui::Checkbox("Chordal parameterization", &data->showCruve6);
			ImGui::Checkbox("Centripetal parameterization", &data->showCruve7);
			ImGui::Checkbox("Foley parameterization", &data->showCruve8);
			ImGui::Text("Creat mode:");
			ImGui::Checkbox("Cubic cruve", &data->cubic);
			ImGui::Checkbox("Bezier cruve", &data->bezier);
			ImGui::Text("Edit mode:");
			ImGui::Checkbox("Edit", &data->edit);
			ImGui::Checkbox("Edit tangent", &data->edit_tan);
			ImGui::Checkbox("Edit tangent line within the scope of G0", &data->g0);
			ImGui::Checkbox("Edit tangent line within the scope of G1", &data->g1);

			// Typically you would use a BeginChild()/EndChild() pair to benefit from a clipping region + own scrolling.
			// Here we demonstrate that this can be replaced by simple offsetting + custom drawing + PushClipRect/PopClipRect() calls.
			// To use a child window instead we could use, e.g:
			//      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));      // Disable padding
			//      ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(50, 50, 50, 255));  // Set a background color
			//      ImGui::BeginChild("canvas", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoMove);
			//      ImGui::PopStyleColor();
			//      ImGui::PopStyleVar();
			//      [...]
			//      ImGui::EndChild();

			// Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
			ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
			ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
			if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
			if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
			ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

			// Draw border and background color
			ImGuiIO& io = ImGui::GetIO();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
			draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

			// This will catch our interactions
			ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool is_hovered = ImGui::IsItemHovered(); // Hovered
			const bool is_active = ImGui::IsItemActive();   // Held
			const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); // Lock scrolled origin
			const pointf2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);

			// Add first and second point
			int edit_index = 0;
			if (is_hovered) {
				float dist = 0;
				int index = 0;
				for (int n = 0; n < data->points.size(); n += 1) {
					dist = pow(pow(data->points[n][0] - mouse_pos_in_canvas[0], 2) + pow(data->points[n][1] - mouse_pos_in_canvas[1], 2), 0.5);
					if (dist < 20) {
						edit_index = n;
					}
					index++;
				}
				

			}
			if (is_hovered && !data->adding_line && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				if (data->edit) {
					data->editing = true;
				}
				else if (data->edit_tan) {
					data->editing_tan = true;
					data->edit_flag = true;
				}
				else {
					data->points.push_back(mouse_pos_in_canvas);
					data->ltangent.push_back(Ubpa::pointf2());
					data->rtangent.push_back(Ubpa::pointf2());
					data->xk.push_back(Slope());
					data->yk.push_back(Slope());
					data->tangent_ratio.push_back(Ratio());
					data->adding_line = true;
				}
			}
			if (data->adding_line)
			{
				data->points.back() = mouse_pos_in_canvas;
				if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
					data->adding_line = false;
				
			}
			if (data->editing) {
				data->points[edit_index] = mouse_pos_in_canvas;
				
				if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
					data->editing = false;
			}
			if (data->editing_tan) {
				data->editing_tan_index = 0;
				for (int i = 0; i < data->points.size(); i++) {
					if (std::abs(data->ltangent[i][0] - mouse_pos_in_canvas[0]) < point_radius + 10
						&& std::abs(data->ltangent[i][1] - mouse_pos_in_canvas[1]) < point_radius + 10) {
						data->editing_tan_index = -(i + 1);

					}
					if (std::abs(data->rtangent[i][0] - mouse_pos_in_canvas[0]) < point_radius + 10
						&& std::abs(data->rtangent[i][1] - mouse_pos_in_canvas[1]) < point_radius + 10) {
						data->editing_tan_index = i + 1;

					}
				}
				if (data->g0) {
					
					if (data->editing_tan_index < 0) {
						data->ltangent[-1 - data->editing_tan_index] = mouse_pos_in_canvas;
					}
					if (data->editing_tan_index > 0) {
						data->rtangent[data->editing_tan_index - 1] = mouse_pos_in_canvas;
					}
					if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
						data->editing_tan = false;
				}
				if (data->g1) {
					if (data->editing_tan_index < 0) {
					
						data->ltangent[-1 - data->editing_tan_index] = mouse_pos_in_canvas;
						float ratio_distance = data->points[-1 - data->editing_tan_index].distance(data->rtangent[-1 - data->editing_tan_index]) /
							data->points[-1 - data->editing_tan_index].distance(mouse_pos_in_canvas);
						float x = mouse_pos_in_canvas[0] - data->points[-1 - data->editing_tan_index][0];
						x = data->points[-1 - data->editing_tan_index][0] - x * ratio_distance;
						float y = mouse_pos_in_canvas[1] - data->points[-1 - data->editing_tan_index][1];
						y = data->points[-1 - data->editing_tan_index][1] - y * ratio_distance;
						data->rtangent[-1 - data->editing_tan_index] = Ubpa::pointf2(x, y);
					
					}
					if (data->editing_tan_index > 0) {
						data->rtangent[data->editing_tan_index - 1] = mouse_pos_in_canvas;
						float ratio_distance = data->points[data->editing_tan_index - 1].distance(data->ltangent[data->editing_tan_index - 1]) /
							data->points[data->editing_tan_index - 1].distance(mouse_pos_in_canvas);
						float x = mouse_pos_in_canvas[0] - data->points[data->editing_tan_index - 1][0];
						x = data->points[data->editing_tan_index - 1][0] - x * ratio_distance;
						float y = mouse_pos_in_canvas[1] - data->points[data->editing_tan_index - 1][1];
						y = data->points[data->editing_tan_index - 1][1] - y * ratio_distance;
						data->ltangent[data->editing_tan_index - 1] = Ubpa::pointf2(x, y);
						
					}
					if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
						data->editing_tan = false;
				}

			}
			
			// Pan (we use a zero mouse threshold when there's no context menu)
			// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan))
			{
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
			}

			// Context menu (under default mouse threshold)
			ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");
			if (ImGui::BeginPopup("context"))
			{
				if (data->adding_line)
					data->points.resize(data->points.size() - 2);
				data->adding_line = false;
				if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0)) { data->points.resize(data->points.size() - 2); }
				if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0)) { data->points.clear(); }
				ImGui::EndPopup();
			}

			// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
			// Draw lines
			if (data->showCruve1)
			{
				int N = data->points.size();
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				int index = 0;
				for (int n = 0; n < N ; n ++) {
					x_veh[index] = data->points[n][0];
					y_veh[index] = data->points[n][1];
					index++;
				}

				// Fit waypoints
				auto coeffs = polyfit(x_veh, y_veh, N - 1);
				double min = find_min_x(x_veh);
				double max = find_max_x(x_veh);
				for (double i = min;i < max;i += 0.1)
				{
					double x = i;
					double y = polyeval(coeffs, i);

					draw_list->AddCircleFilled(ImVec2(origin.x + x, origin.y + y), 2.0f, IM_COL32(255, 0, 0, 255), 12);
				}

			}
			// draw gauss line
			if (data->showCruve2)
			{
				int N = data->points.size();
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				int index = 0;
				for (int n = 0; n < N; n += 1) {
					x_veh[index] = data->points[n][0];
					y_veh[index] = data->points[n][1];
					index++;
				}

				// Fit waypoints
				auto b = gaussfit(x_veh, y_veh, N * 10);
				double min = find_min_x(x_veh);
				double max = find_max_x(x_veh);
				for (double i = min;i < max;i += 0.1)
				{
					double x = i;
					double y = gausseval(b, x_veh, x, N * 10);
					draw_list->AddCircleFilled(ImVec2(origin.x + x, origin.y + y), 2.0f, IM_COL32(0, 0, 255, 255), 12);
				}


			}
			if (data->showCruve3)
			{
				int N = data->points.size();
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				int index = 0;
				for (int n = 0; n < data->points.size(); n += 1) {
					x_veh[index] = data->points[n][0];
					y_veh[index] = data->points[n][1];
					index++;
				}

				// Fit waypoints
				auto coeffs = polyappro(x_veh, y_veh, N - 2);
				double min = find_min_x(x_veh);
				double max = find_max_x(x_veh);
				for (double i = min;i < max;i += 1)
				{
					double y = polyeval(coeffs, i);

					draw_list->AddCircleFilled(ImVec2(origin.x + i, origin.y + y), 2.0f, IM_COL32(0, 255, 0, 255), 12);
				}
			}
			if (data->showCruve4)
			{
				int N = 0;
				for (int n = 0; n < data->points.size(); n += 1) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				int index = 0;
				for (int n = 0; n < data->points.size(); n += 1) {
					x_veh[index] = data->points[n][0];
					y_veh[index] = data->points[n][1];
					index++;
				}

				// Fit waypoints
				auto coeffs = polyappro_op(x_veh, y_veh, N - 2, 5);
				double min = find_min_x(x_veh);
				double max = find_max_x(x_veh);
				for (double i = min;i < max;i += 1)
				{
					double y = polyeval(coeffs, i);

					draw_list->AddCircleFilled(ImVec2(origin.x + i, origin.y + y), 2.0f, IM_COL32(255, 0, 255, 255), 12);
				}

			}
			if (data->showCruve5) {
				// Equidistant
				// Here are the parameters, x and y
				int N = 0;
				for (int n = 0; n < data->points.size(); n += 1) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				VectorXd t_veh(N);
				int index = 0;
				for (int n = 0; n < data->points.size(); n += 1) {
					x_veh[index] = data->points[n][0];
					y_veh[index] = data->points[n][1];
					t_veh[index] = index + 1;
					index++;
				}
				// Fit waypoints
				// auto coeffs_x = polyappro_op(t_veh, x_veh, N - 2, 5);
				// auto coeffs_y = polyappro_op(t_veh, y_veh, N - 2, 5);

				// when we don't use the lambda 
				auto coeffs_x = polyappro(t_veh, x_veh, N - 2);
				auto coeffs_y = polyappro(t_veh, y_veh, N - 2);

				for (double i = 1;i < N; i += 0.01)
				{
					double x = polyeval(coeffs_x, i);
					double y = polyeval(coeffs_y, i);

					draw_list->AddCircleFilled(ImVec2(origin.x + x, origin.y + y), 2.0f, IM_COL32(255, 0, 255, 255), 12);
				}

			}
			if (data->showCruve6) {
				// chordal
				// Here are the parameters, x and y
				int N = 0;
				for (int n = 0; n < data->points.size(); n += 1) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				VectorXd t_veh(N);
				int index = 0;
				// We only need to modify here
				for (int n = 0; n < data->points.size(); n += 1) {
					x_veh[index] = data->points[n][0];
					y_veh[index] = data->points[n][1];
					if (index == 0) {
						t_veh[0] = 0;
					}
					else {
						t_veh[index] = t_veh[index - 1] + pow(pow(x_veh[index] - x_veh[index - 1], 2) + pow(y_veh[index] - y_veh[index - 1], 2), 0.5);
					}

					index++;
				}
				// Fit waypoints
				// auto coeffs_x = polyappro_op(t_veh, x_veh, N - 2, 5);
				// auto coeffs_y = polyappro_op(t_veh, y_veh, N - 2, 5);

				// when we don't use the lambda 
				auto coeffs_x = polyappro(t_veh, x_veh, N - 2);
				auto coeffs_y = polyappro(t_veh, y_veh, N - 2);

				for (double i = 1;i < t_veh[N - 1]; i += 1)
				{
					double x = polyeval(coeffs_x, i);
					double y = polyeval(coeffs_y, i);

					draw_list->AddCircleFilled(ImVec2(origin.x + x, origin.y + y), 2.0f, IM_COL32(255, 0, 150, 255), 12);
				}

			}
			if (data->showCruve7) {
				// chordal
				// Here are the parameters, x and y
				int N = 0;
				for (int n = 0; n < data->points.size(); n += 1) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				VectorXd t_veh(N);
				int index = 0;
				// We only need to modify here
				for (int n = 0; n < data->points.size(); n += 1) {
					x_veh[index] = data->points[n][0];
					y_veh[index] = data->points[n][1];
					if (index == 0) {
						t_veh[0] = 0;
					}
					else {
						t_veh[index] = t_veh[index - 1] + pow(pow(pow(x_veh[index] - x_veh[index - 1], 2) + pow(y_veh[index] - y_veh[index - 1], 2), 0.5), 0.5);
					}

					index++;
				}
				// Fit waypoints
				// auto coeffs_x = polyappro_op(t_veh, x_veh, N - 2, 5);
				// auto coeffs_y = polyappro_op(t_veh, y_veh, N - 2, 5);

				// when we don't use the lambda 
				auto coeffs_x = polyappro(t_veh, x_veh, N - 2);
				auto coeffs_y = polyappro(t_veh, y_veh, N - 2);

				for (double i = 1;i < t_veh[N - 1]; i += 0.1)
				{
					double x = polyeval(coeffs_x, i);
					double y = polyeval(coeffs_y, i);

					draw_list->AddCircleFilled(ImVec2(origin.x + x, origin.y + y), 2.0f, IM_COL32(233, 0, 50, 255), 12);
				}

			}
			if (data->showCruve8) {
				// chordal
				// Here are the parameters, x and y
				int N = 0;
				for (int n = 0; n < data->points.size(); n += 1) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				VectorXd t_veh(N);
				int index = 0;
				// We only need to modify here
				for (int n = 0; n < data->points.size(); n += 1) {
					x_veh[index] = data->points[n][0];
					y_veh[index] = data->points[n][1];

					index++;
				}
				for (int n = 0; n < N; n++) {
					if (n == 0) {
						t_veh[0] = 0;
					}
					else if (n == N - 1 || n == N - 2 || n == N - 3) {
						t_veh[n] = t_veh[n - 1] + pow(pow(x_veh[n] - x_veh[n - 1], 2) + pow(y_veh[n] - y_veh[n - 1], 2), 0.5) * 3;
					}
					else {
						double des1 = pow(pow(x_veh[n] - x_veh[n - 1], 2) + pow(y_veh[n] - y_veh[n - 1], 2), 0.5);        // b
						double des2 = pow(pow(x_veh[n + 1] - x_veh[n], 2) + pow(y_veh[n + 1] - y_veh[n], 2), 0.5);             //c   //b2
						double des3 = pow(pow(x_veh[n + 2] - x_veh[n + 1], 2) + pow(y_veh[n + 2] - y_veh[n + 1], 2), 0.5);      //   c2
						double des4 = pow(pow(x_veh[n + 1] - x_veh[n - 1], 2) + pow(y_veh[n + 1] - y_veh[n - 1], 2), 0.5);   // a
						double des5 = pow(pow(x_veh[n + 2] - x_veh[n], 2) + pow(y_veh[n + 2] - y_veh[n], 2), 0.5);   // a2
						double cos1 = (des1 * des1 + des2 * des2 - des4 * des4) / (2 * des1 * des2);
						double cos2 = (des2 * des2 + des3 * des3 - des5 * des5) / (2 * des2 * des3);
						double alpha1 = min(3.14159 - acos((cos1 > 0.99) ? 0.99 : (cos1 < -0.99) ? -0.99 : cos1), 3.14159 / 2);
						double alpha2 = min(3.14159 - acos((cos2 > 0.99) ? 0.99 : (cos2 < -0.99) ? -0.99 : cos2), 3.14159 / 2);
						t_veh[n] = t_veh[n - 1] + des1 * (1 + 1.5 * (alpha1 * des1 / (des1 + des2)) + 1.5 * (alpha2 * des2 / (des2 + des3)));
					}
				}
				// when we don't use the lambda 
				auto coeffs_x = polyappro(t_veh, x_veh, N - 2);
				auto coeffs_y = polyappro(t_veh, y_veh, N - 2);

				for (double i = 1;i < t_veh[N - 1]; i += 1)
				{
					double x = polyeval(coeffs_x, i);
					double y = polyeval(coeffs_y, i);

					draw_list->AddCircleFilled(ImVec2(origin.x + x, origin.y + y), 2.0f, IM_COL32(203, 100, 10, 255), 12);
				}
			}
			
			// Cubic
			if (data->cubic) {
				// chordal parameterization
				// Here are the parameters, x and y
				int N = data->points.size();
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				VectorXd t_veh(N);
				int index = 0;
				for (int n = 0; n < data->points.size(); n += 1) {
					x_veh[index] = data->points[n][0];
					y_veh[index] = data->points[n][1];
					if (index == 0) {
						t_veh[0] = 0;
					}
					else {
						t_veh[index] = t_veh[index - 1] + pow(pow(pow(x_veh[index] - x_veh[index - 1], 2) + pow(y_veh[index] - y_veh[index - 1], 2), 0.5), 0.5);
					}
					index++;
				}
				size_t n = data->points.size() - 1;
				std::vector<float> u(n);
				std::vector<float> h(n);
				std::vector<float> v_x(n);
				std::vector<float> v_y(n);
				std::vector<float> b_x(n);
				std::vector<float> b_y(n);
				h[0] = t_veh[1] - t_veh[0];
				b_x[0] = 6.f * (x_veh[1] - x_veh[0]) / h[0];
				b_y[0] = 6.f * (y_veh[1] - y_veh[0]) / h[0];
				for (int i = 1; i < n; i++) {
					h[i] = t_veh[i + 1] - t_veh[i];
					u[i] = 2.f * (h[i] + h[i - 1]);
					b_x[i] = 6.f * (x_veh[i + 1] - x_veh[i]) / h[i];
					b_y[i] = 6.f * (y_veh[i + 1] - y_veh[i]) / h[i];
					v_x[i] = b_x[i] - b_x[i - 1];
					v_y[i] = b_y[i] - b_y[i - 1];
				}
				std::vector<float> MX(n + 1, 0.f);
				std::vector<float> MY(n + 1, 0.f);
				for (int i = 2; i < n; i++) {
					b_x[i] = h[i - 1] / u[i - 1];
					b_y[i] = h[i - 1] / u[i - 1];
					u[i] -= h[i - 1] * b_x[i];
				}
				// Ly=V
				for (int i = 2; i < n; i++) {
					v_x[i] -= b_x[i] * v_x[i - 1];
					v_y[i] -= b_y[i] * v_y[i - 1];
				}
				// UM=y
				if (n > 1) {
					MX[n - 1] = v_x[n - 1] / u[n - 1];
					MY[n - 1] = v_y[n - 1] / u[n - 1];
				}
				for (int i = n - 2; i >= 1; i--) {
					MX[i] = (v_x[i] - h[i] * MX[i + 1]) / u[i];
					MY[i] = (v_y[i] - h[i] * MY[i + 1]) / u[i];
				}

				// Caculate the tangent rate of x and y
				for (int i = 0; i < n; i++) {
					float x0 = x_veh[i];
					float x1 = x_veh[i + 1];
					float y0 = y_veh[i];
					float y1 = y_veh[i + 1];
					float dx0, dx1, dy0, dy1;
					if (!data->edit_flag) {
						dx0 = (-h[i]) * (MX[i] * 2 + MX[i + 1]) / 6.f + (x_veh[i + 1] - x_veh[i]) / h[i];
						dx1 = (h[i]) * (MX[i + 1] * 2 + MX[i]) / 6.f + (x_veh[i + 1] - x_veh[i]) / h[i];
						dy0 = (-h[i]) * (MY[i] * 2 + MY[i + 1]) / 6.f + (y_veh[i + 1] - y_veh[i]) / h[i];
						dy1 = (h[i]) * (MY[i + 1] * 2 + MY[i]) / 6.f + (y_veh[i + 1] - y_veh[i]) / h[i];
					}
					else {
						dx0 = data->xk[i].r;
						dx1 = data->xk[i + 1].l;
						dy0 = data->yk[i].r;
						dy1 = data->yk[i + 1].l;
					}
					if (!data->edit_flag) {
						data->xk[i].r = dx0;
						data->yk[i].r = dy0;
						if (i != 0) {
							data->xk[i].l = dx0;
							data->yk[i].l = dy0;
						}
						if (i == N - 1) {
							data->xk[i + 1].l = dx1;
							data->yk[i + 1].l = dy1;
						}
					}
					for (float t = t_veh[i]; t <= t_veh[i + 1]; t += t_step) {
						float x = x0 * h0(t_veh[i], t_veh[i + 1], t) +
							x1 * h1(t_veh[i], t_veh[i + 1], t) +
							dx0 * H0(t_veh[i], t_veh[i + 1], t) +
							dx1 * H1(t_veh[i], t_veh[i + 1], t);
						float y = y0 * h0(t_veh[i], t_veh[i + 1], t) +
							y1 * h1(t_veh[i], t_veh[i + 1], t) +
							dy0 * H0(t_veh[i], t_veh[i + 1], t) +
							dy1 * H1(t_veh[i], t_veh[i + 1], t);
						draw_list->AddCircleFilled(ImVec2(origin.x + x, origin.y + y), 2.0f, IM_COL32(233, 0, 50, 255), 12);
					}

				}
				if (data->g0 || data->g1) {
					if(!data->edit_flag){
						// Caculate tangent rate
						for (int i = 0; i < data->points.size() - 1; i++) {
							Ubpa::pointf2 temp = Ubpa::pointf2(data->xk[i].r, data->yk[i].r);
							data->tangent_ratio[i].r = base_tangent_len / temp.distance(Ubpa::pointf2(0.f, 0.f));
							data->rtangent[i] = Ubpa::pointf2(data->points[i][0] + data->xk[i].r * data->tangent_ratio[i].r,
								data->points[i][1] + data->yk[i].r * data->tangent_ratio[i].r);
						}
						for (int i = data->points.size() - 1; i > 0; i--) {
							Ubpa::pointf2 temp = Ubpa::pointf2(data->xk[i].l, data->yk[i].l);
							data->tangent_ratio[i].l = base_tangent_len / temp.distance(Ubpa::pointf2(0.f, 0.f));
							data->ltangent[i] = Ubpa::pointf2(data->points[i][0] - data->xk[i].l * data->tangent_ratio[i].l,
								data->points[i][1] - data->yk[i].l * data->tangent_ratio[i].l);
						}
					}
					
					// draw tangent lines and points
					for (int i = 0; i < data->points.size() - 1; i++) {
						const ImVec2 p1(origin.x + data->points[i][0], origin.y + data->points[i][1]);
						const ImVec2 p2(origin.x + data->rtangent[i][0], origin.y + data->rtangent[i][1]);
						data->xk[i].r = data->rtangent[i][0] - data->points[i][0];
						data->xk[i].r /= data->tangent_ratio[i].r;
						data->yk[i].r = data->rtangent[i][1] - data->points[i][1];
						data->yk[i].r /= data->tangent_ratio[i].r;
						draw_list->AddLine(p1, p2, slope_col, 2.f);
						draw_list->AddCircleFilled(p2, point_radius, normal_point_col);
					}
					for (int i = data->points.size() - 1; i > 0; i--) {
						const ImVec2 p1(origin.x + data->points[i][0], origin.y + data->points[i][1]);
						const ImVec2 p2(origin.x + data->ltangent[i][0], origin.y + data->ltangent[i][1]);
						data->xk[i].l = data->points[i][0] - data->ltangent[i][0];
						data->xk[i].l /= data->tangent_ratio[i].l;
						data->yk[i].l = data->points[i][1] - data->ltangent[i][1];
						data->yk[i].l /= data->tangent_ratio[i].l;
						draw_list->AddLine(p1, p2, slope_col, 2.f);
						draw_list->AddCircleFilled(p2, point_radius, normal_point_col);
					}
				}
			}
			if (data->bezier) {
				size_t n = data->points.size();
				VectorXd x_veh(n);
				VectorXd y_veh(n);
				VectorXd t_veh(n);
				std::vector<Ubpa::pointf2> rtangent(n);
				std::vector<Ubpa::pointf2> ltangent(n);
				int index = 0;
				for (int n1 = 0; n1 < data->points.size(); n1 += 1) {
					x_veh[index] = data->points[n1][0];
					y_veh[index] = data->points[n1][1];
					if (index == 0) {
						t_veh[0] = 0;
					}
					else {
						t_veh[index] = t_veh[index - 1] + pow(pow(pow(x_veh[index] - x_veh[index - 1], 2) + pow(y_veh[index] - y_veh[index - 1], 2), 0.5), 0.5);
					}
					index++;
				}
				if (n == 2) {
					draw_list->AddLine(ImVec2(origin.x + x_veh[0], origin.y + y_veh[0]),
						ImVec2(origin.x + x_veh[1], origin.y + y_veh[1]), IM_COL32(233, 150, 50, 255),12);
					return;
				}
				for (int i = 1; i < n - 1; i++) {
					float dx = x_veh[i + 1] - x_veh[i - 1];
					float dy = y_veh[i + 1] - y_veh[i - 1];
					rtangent[i] = Ubpa::pointf2(x_veh[i] + dx / 6.f, y_veh[i] + dy / 6.f);
					ltangent[i] = Ubpa::pointf2(x_veh[i] - dx / 6.f, y_veh[i] - dy / 6.f);
				}
				rtangent[0] = Ubpa::pointf2(x_veh[0] + (x_veh[1] - x_veh[2]) / 6.f, y_veh[0] + (y_veh[1] - y_veh[2]) / 6.f);
				ltangent[n - 1] = Ubpa::pointf2(x_veh[n - 1] - (x_veh[n - 2] - x_veh[n - 3]) / 6.f, y_veh[n - 1] - (y_veh[n - 2] - y_veh[n - 3]) / 6.f);
				for (int i = 1; i < n ; i++) {
					Ubpa::pointf2 control_points[4] = { data->points[i-1], rtangent[i-1], ltangent[i], data->points[i] };
					std::vector<Ubpa::pointf2> p;
					DeCasteljau(p, control_points);
					for (int j = 0; j < p.size() - 1 ; j++)
						draw_list->AddLine(ImVec2(origin.x + p[j][0], origin.y + p[j][1]),
							ImVec2(origin.x + p[j + 1][0], origin.y + p[j + 1][1]), IM_COL32(233, 150, 50, 255), 4);
				}

			}
			if (data->opt_enable_grid)
			{
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
			}
			for (int n = 0; n < data->points.size(); n += 1)
				draw_list->AddCircleFilled(ImVec2(origin.x + data->points[n][0], origin.y + data->points[n][1]), 2.0f, IM_COL32(255, 255, 0, 255), 12);
			draw_list->PopClipRect();
		}

		ImGui::End();
	});
}
