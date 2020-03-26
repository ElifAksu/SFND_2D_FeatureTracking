/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include <deque>


using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    std::ofstream myfile("comparison.csv",std::ios_base::app | std::ios_base::out);
   
    //myfile << "Detector Type,  Detector Time, Detected KeyPoints, Descriptor Type, Descriptor Time, MatcherType, SelectorType, Match Time, Matched KeyPoints\n";
 
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    std::deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);
        if(dataBuffer.size()>dataBufferSize)
        {
            dataBuffer.pop_front();
        }

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        //string detectorType = "SHITOMASI";
        //string detectorType = "HARIS";
        //string detectorType = "FAST";
        //string detectorType = "BRISK";
        //string detectorType = "ORB";
        //string detectorType = "AKAZE";
        string detectorType = "SIFT";

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        double t = (double)cv::getTickCount();

        myfile << detectorType.c_str()<<",";
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if(detectorType.compare("HARIS") == 0)
        {
           detKeypointsHarris(keypoints, imgGray,false);
        }
        else
        {
          detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        myfile << (1000 * t / 1.0)<<",";
       
        
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
       
        if (bFocusOnVehicle)
        {
             vector<cv::KeyPoint> keypoints_bbox;
           
           for(auto kp = keypoints.begin(); kp <= keypoints.end(); kp++)
           {
               if(vehicleRect.contains(kp->pt))
               {
                keypoints_bbox.push_back(*kp);
               }
           }
           keypoints=keypoints_bbox;
        }

        //// EOF STUDENT ASSIGNMENT
        std::cout<< "keypoints on the vehicle : " << keypoints.size() <<std::endl;
        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;
        myfile << keypoints.size()<<",";

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        //string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        //string descriptorType = "BRIEF";
        //string descriptorType = "ORB";
        //string descriptorType = "FREAK";
        //string descriptorType = "AKAZE";
        string descriptorType = "SIFT";

        t = (double)cv::getTickCount();
        myfile << descriptorType.c_str() << ",";
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        myfile << (1000 * t / 1.0)<<",";
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            //string matcherType = "MAT_FLANN"; 

            string descriptor_ ; //= "DES_BINARY"; // DES_BINARY, DES_HOG
            if(descriptorType.compare("SIFT") == 0)
            {
                descriptor_ = "DES_HOG";
            }
            else
            {
                descriptor_ = "DES_BINARY";
            }
            

            string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN
            //string selectorType = "SEL_KNN";  

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
            double match_t = (double)cv::getTickCount();
            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptor_, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT
            myfile << matcherType.c_str()<<"," ;
            myfile << selectorType.c_str()<<"," ;
            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;
            match_t = ((double)cv::getTickCount() - match_t) / cv::getTickFrequency();
            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;
            cout <<  " match Descriptors run  " << 1000 * match_t/ 1.0 << " ms" << endl;
            cout <<  " matches number   " << matches.size() << endl;
            
            myfile << (1000 * match_t / 1.0)<<",";

            myfile << matches.size() << "\n";
            // visualize matches between current and previous image
            bVis = false; //true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    return 0;
}
