#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include <Eigen/Core>

#include <Eigen/QR>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace Ubpa;
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
			ImGui::Text("！！！！！！！！！！！！");
			ImGui::Checkbox("Cubic cruve", &data->cubic);


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
			if (is_hovered && !data->adding_line && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				data->points.push_back(mouse_pos_in_canvas);
				data->points.push_back(mouse_pos_in_canvas);
				data->adding_line = true;
			}
			if (data->adding_line)
			{
				data->points.back() = mouse_pos_in_canvas;
				if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
					data->adding_line = false;
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
			double S(VectorXd m, VectorXd h, VectorXd var, VectorXd T, int segment, double t) {
				double res = m[segment] * pow(T[i + 1] - t, 3) / (6 * h[segment])
					+ m[segment + 1] * pow(t - T[i], 3) / (6 * h[segment])
					+ (var[i + 1] / h[i] - m[i + 1] * h[i] / 6) * (t - T[i])
					+ (var[i] / h[i] - m[i] * h[i] / 6) * (T[i + 1] - t);
				return res;
			}
			
			if (data->opt_enable_grid)
			{
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
			}
			for (int n = 0; n < data->points.size(); n += 2)
				draw_list->AddCircleFilled(ImVec2(origin.x + data->points[n + 1][0], origin.y + data->points[n + 1][1]), 2.0f, IM_COL32(255, 255, 0, 255), 12);
			draw_list->PopClipRect();
		}

		ImGui::End();
	});
}
