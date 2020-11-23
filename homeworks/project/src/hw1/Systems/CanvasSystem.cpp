#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include <Eigen/Core>

#include <Eigen/QR>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace Ubpa;
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
double hermite(int N, double x, Eigen::VectorXd xi, Eigen::VectorXd yi, Eigen::VectorXd dyi)
{
	int i, j;
	double li, sum, y;
	Eigen::VectorXd gix(N);
	Eigen::VectorXd hix(N);
	for (i = 0;i < N;i++)
	{
		li = 1.0; sum = 0.0;
		for (j = 0;j < N;j++)
			if (j != i)
			{
				li = li * (x - xi[j]) / (xi[i] - xi[j]);   
				sum = sum + 1.0 / (xi[i] - xi[j]);
			}
		li = li * li;
		gix[i] = (1.0 - 2.0 * (x - xi[i]) * sum) * li;
		hix[i] = (x - xi[i]) * li;
	}
	y = 0.0;
	for (i = 0;i < N;i++) y = y + yi[i] * gix[i] + dyi[i] * hix[i];
	return y;
}
void CubicSpline(std::vector<Ubpa::pointf2>&, std::vector<Ubpa::pointf2>&, std::vector<Slope>&, bool);
void SlopeSpline(std::vector<Ubpa::pointf2>&, std::vector<Ubpa::pointf2>&, std::vector<Slope>&);
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
			ImGui::Checkbox("Hermit cruve", &data->hermit);
			ImGui::Text("Edit mode:");
			ImGui::Checkbox("Edit", &data->edit);
			static float m_mul = 1.0f;
			static float h_mul = 0.0f;
			ImGui::SliderFloat("direction", &m_mul, -1.0f, 1.0f);
			ImGui::SliderFloat("length", &h_mul, -10.0f, 10.0f);


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
			
			float dist = 0;
			int index = 0;
			for (int n = 0; n < data->points.size(); n += 2) {
				dist = pow(pow(data->points[n + 1][0] - mouse_pos_in_canvas[0], 2) + pow(data->points[n + 1][1] - mouse_pos_in_canvas[1], 2), 0.5);
				if (dist < 20) {
					data->edit_index = n + 1;
					m_mul = 1.0f;
					h_mul = 0.0f;
				}
				index++;
			}

			
			
			if (is_hovered && !data->adding_line && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				if (data->edit) {
					data->editing = true;
				}
				else {
					data->points.push_back(mouse_pos_in_canvas);
					data->points.push_back(mouse_pos_in_canvas);
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
				data->points[data->edit_index] = mouse_pos_in_canvas;
				if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
					data->editing = false;
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
				int N = 0;
				for (int n = 0; n < data->points.size(); n += 2) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				int index = 0;
				for (int n = 0; n < data->points.size(); n += 2) {
					x_veh[index] = data->points[n + 1][0];
					y_veh[index] = data->points[n + 1][1];
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
				int N = 0;
				for (int n = 0; n < data->points.size(); n += 2) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				int index = 0;
				for (int n = 0; n < data->points.size(); n += 2) {
					x_veh[index] = data->points[n + 1][0];
					y_veh[index] = data->points[n + 1][1];
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
				int N = 0;
				for (int n = 0; n < data->points.size(); n += 2) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				int index = 0;
				for (int n = 0; n < data->points.size(); n += 2) {
					x_veh[index] = data->points[n + 1][0];
					y_veh[index] = data->points[n + 1][1];
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
				for (int n = 0; n < data->points.size(); n += 2) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				int index = 0;
				for (int n = 0; n < data->points.size(); n += 2) {
					x_veh[index] = data->points[n + 1][0];
					y_veh[index] = data->points[n + 1][1];
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
				for (int n = 0; n < data->points.size(); n += 2) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				VectorXd t_veh(N);
				int index = 0;
				for (int n = 0; n < data->points.size(); n += 2) {
					x_veh[index] = data->points[n + 1][0];
					y_veh[index] = data->points[n + 1][1];
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
				for (int n = 0; n < data->points.size(); n += 2) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				VectorXd t_veh(N);
				int index = 0;
				// We only need to modify here
				for (int n = 0; n < data->points.size(); n += 2) {
					x_veh[index] = data->points[n + 1][0];
					y_veh[index] = data->points[n + 1][1];
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
				for (int n = 0; n < data->points.size(); n += 2) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				VectorXd t_veh(N);
				int index = 0;
				// We only need to modify here
				for (int n = 0; n < data->points.size(); n += 2) {
					x_veh[index] = data->points[n + 1][0];
					y_veh[index] = data->points[n + 1][1];
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
				for (int n = 0; n < data->points.size(); n += 2) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				VectorXd t_veh(N);
				int index = 0;
				// We only need to modify here
				for (int n = 0; n < data->points.size(); n += 2) {
					x_veh[index] = data->points[n + 1][0];
					y_veh[index] = data->points[n + 1][1];

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
				int N = 0;
				for (int n = 0; n < data->points.size(); n += 2) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				VectorXd t_veh(N);
				int index = 0;
				for (int n = 0; n < data->points.size(); n += 2) {
					x_veh[index] = data->points[n + 1][0];
					y_veh[index] = data->points[n + 1][1];
					if (index == 0) {
						t_veh[0] = 0;
					}
					else {
						t_veh[index] = t_veh[index - 1] + pow(pow(pow(x_veh[index] - x_veh[index - 1], 2) + pow(y_veh[index] - y_veh[index - 1], 2), 0.5), 0.5);
					}
					index++;
				}
				if (data->edit) {
					t_veh[data->edit_index/2] *= m_mul;
					t_veh[data->edit_index/2] += h_mul;
				}
				Eigen::VectorXd h(N - 1);
				Eigen::VectorXd u(N - 1);
				Eigen::VectorXd b_x(N - 1);
				Eigen::VectorXd b_y(N - 1);
				Eigen::VectorXd v_x(N - 1);
				Eigen::VectorXd v_y(N - 1);
				for (int i = 0; i < N - 1;i++) {
					h[i] = t_veh[i + 1] - t_veh[i];
					b_x[i] = 6 * (x_veh[i + 1] - x_veh[i]) / h[i];
					b_y[i] = 6 * (y_veh[i + 1] - y_veh[i]) / h[i];
				}
				Eigen::VectorXd h1 = h;
				for (int i = 1; i < N - 1;i++) {
					u[i-1] = 2 * (h[i] + h[i - 1]);
					v_x[i-1] = b_x[i] - b_x[i - 1];
					v_y[i-1] = b_y[i] - b_y[i - 1];
				}
				h1[0] = h1[0] / u[0];
				v_x[0] = v_x[0] / u[0];
				v_y[0] = v_y[0] / u[0];

				for (int i = 1; i < N-1; i++) {
					float m1 = 1.0f / (u[i] - h1[i] * h1[i - 1]);
					h1[i] = h1[i] * m1;
					v_x[i] = (v_x[i] - h1[i] * v_x[i - 1]) * m1;
					v_y[i] = (v_y[i] - h1[i] * v_y[i - 1]) * m1;
				}

				for (int i = N - 2; i-- > 0; ) {
					v_x[i] = v_x[i] - h1[i] * v_x[i + 1];
					v_y[i] = v_y[i] - h1[i] * v_y[i + 1];
				}
				VectorXd m_x1(N + 1);
				VectorXd m_y1(N + 1);
				for (int i = 1; i < N;i++) {
					m_x1[i] = v_x[i-1];
					m_y1[i] = v_y[i-1];
				}
				m_x1[0] = 0;
				m_y1[0] = 0;
				m_x1[N] = 0;
				m_y1[N] = 0;
				int segment = 0;
				
				for (double i = 1;i < t_veh[N-1]; i += 0.1)
				{
					if (i > t_veh[segment+1]) {
						segment += 1;
					}
					if (segment >= N - 1) {
						double x = S_1(m_x1, h, x_veh, t_veh, segment, i);
						double y = S_1(m_y1, h, y_veh, t_veh, segment, i);
						draw_list->AddCircleFilled(ImVec2(origin.x + x, origin.y + y), 2.0f, IM_COL32(233, 0, 50, 255), 12);
						break;
					}
					double x = S_1(m_x1, h, x_veh, t_veh, segment, i);
					double y = S_1(m_y1, h, y_veh, t_veh, segment, i);

					draw_list->AddCircleFilled(ImVec2(origin.x + x, origin.y + y), 2.0f, IM_COL32(233, 0, 50, 255), 12);
				}
			}
			if (data->hermit) {
				// chordal parameterization
				// Here are the parameters, x and y
				int N = 0;
				for (int n = 0; n < data->points.size(); n += 2) {
					N++;
				}
				VectorXd x_veh(N);
				VectorXd y_veh(N);
				VectorXd t_veh(N);
				int index = 0;
				for (int n = 0; n < data->points.size(); n += 2) {
					x_veh[index] = data->points[n + 1][0];
					y_veh[index] = data->points[n + 1][1];
					if (index == 0) {
						t_veh[0] = 0;
					}
					else {
						t_veh[index] = t_veh[index - 1] + pow(pow(pow(x_veh[index] - x_veh[index - 1], 2) + pow(y_veh[index] - y_veh[index - 1], 2), 0.5), 0.5);
					}
					index++;
				}
				Eigen::VectorXd h(N - 1);
				Eigen::VectorXd u(N - 1);
				Eigen::VectorXd b_x(N - 1);
				Eigen::VectorXd b_y(N - 1);
				Eigen::VectorXd v_x(N - 1);
				Eigen::VectorXd v_y(N - 1);
				for (int i = 0; i < N - 1;i++) {
					h[i] = t_veh[i + 1] - t_veh[i];
					b_x[i] = 6 * (x_veh[i + 1] - x_veh[i]) / h[i];
					b_y[i] = 6 * (y_veh[i + 1] - y_veh[i]) / h[i];
				}
				Eigen::VectorXd h1 = h;
				for (int i = 1; i < N - 1;i++) {
					u[i - 1] = 2 * (h[i] + h[i - 1]);
					v_x[i - 1] = b_x[i] - b_x[i - 1];
					v_y[i - 1] = b_y[i] - b_y[i - 1];
				}
				h1[0] = h1[0] / u[0];
				v_x[0] = v_x[0] / u[0];
				v_y[0] = v_y[0] / u[0];

				for (int i = 1; i < N - 1; i++) {
					float m1 = 1.0f / (u[i] - h1[i] * h1[i - 1]);
					h1[i] = h1[i] * m1;
					v_x[i] = (v_x[i] - h1[i] * v_x[i - 1]) * m1;
					v_y[i] = (v_y[i] - h1[i] * v_y[i - 1]) * m1;
				}

				for (int i = N - 2; i-- > 0; ) {
					v_x[i] = v_x[i] - h1[i] * v_x[i + 1];
					v_y[i] = v_y[i] - h1[i] * v_y[i + 1];
				}
				VectorXd m_x1(N + 1);
				VectorXd m_y1(N + 1);
				for (int i = 1; i < N;i++) {
					m_x1[i] = v_x[i - 1];
					m_y1[i] = v_y[i - 1];
				}
				m_x1[0] = 0;
				m_y1[0] = 0;
				m_x1[N] = 0;
				m_y1[N] = 0;
				VectorXd diff_x(N);
				VectorXd diff_y(N);
				for (int i = 0; i < N-1; i++) {
					// differential coefficient 
					diff_x[i] = h[i] * m_x1[i] / 6 + h[i] * m_x1[i + 1] / 3 - x_veh[i] / h[i] + x_veh[i + 1] / h[i];
					diff_x[i] = h[i] * m_y1[i] / 6 + h[i] * m_y1[i + 1] / 3 - y_veh[i] / h[i] + y_veh[i + 1] / h[i];
				}
				diff_x[N - 1] = 0;
				diff_y[N - 1] = 0;
				for (double i = 1;i < t_veh[N - 1]; i += 0.1)
				{
					double x = hermite(N,i, t_veh, x_veh, diff_x);
					double y = hermite(N,i, t_veh, y_veh, diff_y);

					draw_list->AddCircleFilled(ImVec2(origin.x + x, origin.y + y), 2.0f, IM_COL32(50, 0, 255, 255), 12);
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
			if (!data->enable_add_point) {
				if (data->edit_point == 0) {
					if (ImGui::MenuItem("Edit Points' position", NULL, false, data->points.size() > 0))
						data->edit_point = 1;
					if (data->fitting_type == 0) {
						if (ImGui::MenuItem("Edit Points' slope (G0)", NULL, false))
							data->edit_point = 2;
						if (ImGui::MenuItem("Edit Points' slope (G1)", NULL, false))
							data->edit_point = 3;
					}
				}
				else {
					if (ImGui::MenuItem("Cancel Edit Points", NULL, false, data->points.size() > 0))
						data->edit_point = 0;
				}
				if (ImGui::MenuItem("Enable Add Points", NULL, false))
					data->enable_add_point = true, data->edit_point = 0;
			}
			for (int n = 0; n < data->points.size(); n += 2) {
				
				if (n+1 == data->edit_index) {
					draw_list->AddCircleFilled(ImVec2(origin.x + data->points[n + 1][0], origin.y + data->points[n + 1][1]), 4.0f, IM_COL32(0, 0, 255, 255), 12);
				}
				else {
					draw_list->AddCircleFilled(ImVec2(origin.x + data->points[n + 1][0], origin.y + data->points[n + 1][1]), 2.0f, IM_COL32(255, 255, 0, 255), 12);

				}
				 
				
			}
				
			draw_list->PopClipRect();
		}

		ImGui::End();
	});
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
void SlopeSpline(std::vector<Ubpa::pointf2>& ret, std::vector<Ubpa::pointf2>& p, std::vector<Slope>& k) {
	//spdlog::info("SlopeSpline");
	for (int i = 0; i < p.size() - 1; i++) {
		float y0 = p[i][1];
		float y1 = p[i + 1][1];
		float dy0 = k[i].r;
		float dy1 = k[i + 1].l;
		for (float x = p[i][0]; x <= p[i + 1][0]; x += t_step) {
			//S = h0*y0 + h1*y1 + dy0*H0 + dy1*H1;
			ret.push_back(Ubpa::pointf2(x,
				y0 * h0(p[i][0], p[i + 1][0], x) +
				y1 * h1(p[i][0], p[i + 1][0], x) +
				dy0 * H0(p[i][0], p[i + 1][0], x) +
				dy1 * H1(p[i][0], p[i + 1][0], x)));
		}
	}
}