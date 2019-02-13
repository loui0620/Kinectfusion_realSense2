#pragma warning( disable : 4996 )
#include "Header.h"

using namespace cv;
using namespace std;
void register_glfw_callbacks(window &app, glfw_state& app_state);
void make_depth_histogram(const cv::Mat &depth, cv::Mat &normalized_depth);
void saveRawBinFiles(const string& folderName, const vector<Mat>& matrices)
{
	vector<int> compression_params;
	//compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);

	const char *cFolderName = folderName.c_str();
	for (int i = 0; i < matrices.size(); ++i)
	{
		Mat normalized;
		make_depth_histogram(matrices.at(i), normalized);
		string savePath = folderName + "/depth/" + to_string(i) + ".png";
		string normalPath = folderName + "/debug/" + to_string(i) + ".png";
		cv::Mat mat = matrices.at(i);
		imwrite(savePath, mat);
		imwrite(normalPath, normalized);
		printf("%i.png saved.\n", i);
	}
	printf("SAVING DONE.");
}

void vecmatwrite(const string& filename, const vector<Mat>& matrices)
{
	ofstream fs(filename, fstream::binary);

	for (size_t i = 0; i < matrices.size(); ++i)
	{
		const Mat& mat = matrices[i];

		// Header
		int type = mat.type();
		int channels = mat.channels();
		fs.write((char*)&mat.rows, sizeof(int));    // rows
		fs.write((char*)&mat.cols, sizeof(int));    // cols
		fs.write((char*)&type, sizeof(int));        // type
		fs.write((char*)&channels, sizeof(int));    // channels

													// Data
		if (mat.isContinuous())
		{
			fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
		}
		else
		{
			int rowsz = CV_ELEM_SIZE(type) * mat.cols;
			for (int r = 0; r < mat.rows; ++r)
			{
				fs.write(mat.ptr<char>(r), rowsz);
			}
		}
	}
}

vector<Mat> vecmatread(const string& filename)
{
	vector<Mat> matrices;
	ifstream fs(filename, fstream::binary);

	// Get length of file
	fs.seekg(0, fs.end);
	int length = fs.tellg();
	fs.seekg(0, fs.beg);

	while (fs.tellg() < length)
	{
		// Header
		int rows, cols, type, channels;
		fs.read((char*)&rows, sizeof(int));         // rows
		fs.read((char*)&cols, sizeof(int));         // cols
		fs.read((char*)&type, sizeof(int));         // type
		fs.read((char*)&channels, sizeof(int));     // channels

													// Data
		Mat mat(rows, cols, type);
		fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

		matrices.push_back(mat);
	}
	return matrices;
}

void readDisplayDepth(const std::string &fileName)
{
	std::vector<cv::Mat> sequence;
	int cnt = 0;
	sequence = vecmatread(fileName);
	std::cout << "Input Frame Total: " << sequence.size() << std::endl;

	const auto window_name = "Re-Display Image";
	namedWindow(window_name, WINDOW_AUTOSIZE);
	cv::Mat display;
	//sequence[cnt].copyTo(display);
	cv::waitKey(100);
	sequence.at(0).copyTo(display);
	
	imshow(window_name, display);
	while (cnt < sequence.size() && cvGetWindowHandle(window_name))
	{
		cv::Mat display;
		sequence.at(cnt).copyTo(display);
		cv::waitKey(10);
		imshow(window_name, display);
		cnt++;
		cout << cnt << std::endl;
	}
	cout << "DEBUG" << endl;
}

void make_depth_histogram(const cv::Mat &depth, cv::Mat &normalized_depth) {
	normalized_depth = cv::Mat(depth.size(), CV_8UC1);
	int width = depth.cols, height = depth.rows;

	static uint32_t histogram[0x10000];
	memset(histogram, 0, sizeof(histogram));

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			++histogram[depth.at<ushort>(i, j)];
		}
	}

	for (int i = 2; i < 0x10000; ++i) histogram[i] += histogram[i - 1]; // Build a cumulative histogram for the indices in [1,0xFFFF]

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (uint16_t d = depth.at<ushort>(i, j)) {
				int f = histogram[d] * 255 / histogram[0xFFFF]; // 0-255 based on histogram location
				normalized_depth.at<uchar>(i, j) = static_cast<uchar>(f);
			}
			else {
				normalized_depth.at<uchar>(i, j) = 0;
			}
		}
	}
}

int main(int argv, char *arv[]) try {
	
	texture depth_img, color_img;
	std::vector<cv::Mat> frameSequal;
	cv::Size depthSize = cv::Size(640, 480);

	//readDisplayDepth("test.bin");
	
	int frameCount = 1;
	// Declare depth colorizer for pretty visualization of depth data
	rs2::colorizer color_map;

	// Declare RealSense pipeline, encapsulating the actual device and sensors
	rs2::pipeline pipe;
	rs2::config cfg;
	
	// Set output depth as original resolution
	cfg.enable_stream(RS2_STREAM_DEPTH, -1, depthSize.width, depthSize.height, rs2_format::RS2_FORMAT_ANY, 0);

	// Start streaming with default recommended configuration
	rs2::pipeline_profile profile = pipe.start(cfg);
	
	// Set RS2 as High-Accuracy Mode
	auto sensor = profile.get_device().first<rs2::depth_sensor>();
	sensor.set_option(rs2_option::RS2_OPTION_VISUAL_PRESET, rs2_rs400_visual_preset::RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY);

	// Set pre-processing filter
	rs2::decimation_filter dec_filter;
	rs2::spatial_filter spa_filter;
	dec_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 3);
	const auto window_name = "Display Image";
	namedWindow(window_name, WINDOW_AUTOSIZE);

	
	while (waitKey(1) < 0 && cvGetWindowHandle(window_name))
	{
		
		rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
		rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
		rs2::frame rawDepth = data.get_depth_frame();
		// Query frame size (width and height)
		const int w = depth.as<rs2::video_frame>().get_width();
		const int h = depth.as<rs2::video_frame>().get_height();

		// Create OpenCV matrix of size (w,h) from the colorized depth data
		Mat image(Size(w, h), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);
		Mat rawMat(cv::Size(w, h), CV_16U, (void*)rawDepth.get_data(), Mat::AUTO_STEP);
		
		//DEBUG
		std::string sequalStamp = std::to_string(frameCount);
		cv::putText(image,
			sequalStamp,
			cv::Point(25, 25), // Coordinates
			cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
			1.0, // Scale. 2.0 = 2x bigger
			cv::Scalar(255, 255, 255), // BGR Color
			1); // Anti-alias (Optional)

		// Normalized reference frame in 0-255 range.
		Mat normalized;
		make_depth_histogram(rawMat, normalized);
		
		// Save raw depth in mm-unit.
		frameSequal.push_back(rawMat);
		// Update the window with new data
		imshow(window_name, image);
		frameCount++;
		printf("%i\n", frameCount);
		if (frameCount > 200)
		{
			break;
		}
	}

	//vecmatwrite("test.bin", frameSequal);
	saveRawBinFiles("D:/dataset/output_imgs", frameSequal);
	printf("Written Total: %i\n", frameSequal.size());
	//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
	
	return EXIT_SUCCESS;
	
}

catch (const rs2::error & e)
{
	std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}