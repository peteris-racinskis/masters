#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <ros/ros.h>
#include<cmath>
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Pose.h"
#include <rviz_visual_tools/rviz_visual_tools.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
using namespace std;

namespace defaults {
	const string FNAME = "/home/user/repos/masters/processed_data/train_datasets/train-start-time-doubled-5e9156387f59cb9efb35.csv";
	const int REPEATS = 1;
	const int START = 1;
	const int STOP = 51;
	const float OFFSET = 0.6;
	const float CUBE_SIZE = 0.03;
}

int main(int argc, char **argv)
{
	std::vector<std::string> args;
	string fname;
	int repeats;
	int stop, start;

	if (argc > 1) {
		args.assign(argv + 1, argv + argc);
	}

	fname = (args.size() > 0) ? args[0] : defaults::FNAME;
	repeats = (args.size() > 1) ? stoi(args[1]) : defaults::REPEATS;
	stop = (args.size() > 2) ? stoi(args[2]) : defaults::STOP;
	start = (args.size() > 3) ? stoi(args[3]) : defaults::START;

	ros::init(argc, argv,"trajectory_visualizer");
	ros::NodeHandle node_obj;
	moveit_visual_tools::MoveItVisualToolsPtr moveit_visual_tools_;
	moveit_visual_tools_.reset(new moveit_visual_tools::MoveItVisualTools("panda_link0","/moveit_visual_markers"));
	moveit_visual_tools_->deleteAllMarkers();
 
	vector<vector<string>> content;
	vector<rviz_visual_tools::colors> colors;
	vector<string> row;
	float length = 0.0;
	geometry_msgs::Pose pose, target_pose;
	string line, word;

	colors.push_back(rviz_visual_tools::colors::GREEN);
	colors.push_back(rviz_visual_tools::colors::YELLOW);
	colors.push_back(rviz_visual_tools::colors::RED);
	colors.push_back(rviz_visual_tools::colors::WHITE);

	fstream file (fname, ios::in);
	if(file.is_open())
	{
		while(getline(file, line))
		{
			row.clear();
 
			stringstream str(line);

			while(getline(str, word, ','))
				row.push_back(word);
			content.push_back(row);
		}
	}
	else
		cout<<"Could not open the file\n";

	
	int new_stop = stop;
	for(int j=0;j<repeats;j++){


		if (new_stop > content.size()){
			cout << "Size exceeded! Size: " << content.size() << " Stop index: " << new_stop << endl;
			break;
		}

		cout << "Start: " << start << " Stop: " << new_stop << endl;

		int end = int(content[start].size());

		target_pose.orientation.w = 1;
		target_pose.orientation.x = 0;
		target_pose.orientation.y = 0;
		target_pose.orientation.z = 0;

		target_pose.position.x = 0;
		target_pose.position.y = -j * defaults::OFFSET;
		target_pose.position.z = -0.2;

		// publish a cube at the root of the trajectory for id by color
		moveit_visual_tools_->publishCuboid(target_pose, defaults::CUBE_SIZE, defaults::CUBE_SIZE, defaults::CUBE_SIZE, colors[j % 4]);

		target_pose.position.x += stof(content[start][end-4]);
		target_pose.position.y += stof(content[start][end-3]);
		target_pose.position.z += 0.2 + stof(content[start][end-2]);

		// publish a cube at the target position
		moveit_visual_tools_->publishCuboid(target_pose, defaults::CUBE_SIZE, defaults::CUBE_SIZE, defaults::CUBE_SIZE, colors[j % 4]);

		for(int i=start;i<new_stop;i++)
		{
			pose.position.x = stof(content[i][1]);
			pose.position.y = stof(content[i][2]) - j * defaults::OFFSET;
			pose.position.z = stof(content[i][3]);

			length = pow(stof(content[i][4]),2) + pow(stof(content[i][5]),2) + pow(stof(content[i][6]),2) + pow(stof(content[i][7]),2);
			length  = sqrt(length);

			pose.orientation.x = stof(content[i][4])/length;
			pose.orientation.y = stof(content[i][5])/length;
			pose.orientation.z = stof(content[i][6])/length;
			pose.orientation.w = stof(content[i][7])/length;

			rviz_visual_tools::scales scale = (stof(content[i][8]) > 0.95) ? rviz_visual_tools::scales::XXXSMALL : rviz_visual_tools::scales::XSMALL;

			moveit_visual_tools_->publishAxis(pose,scale);
			moveit_visual_tools_->trigger();
		}
		

		start += stop;
		new_stop += stop;
	}
	return 0;
}