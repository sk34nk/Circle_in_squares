#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

uint8_t thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";

const String keys =
{
    "{help      |       | print this message    }"
    "{@file     |1.png  | image for find        }"
    "{number    |0      | inclined squares      }"
    "{path      |       | path to file          }"
};

void printHelp( void )
{
    cout << " Usage: Circle_in_square -help file.png -number=<inclined>"
            " -path=<pathToFile> " << endl;
}

static double angle(Point pt1, Point pt2, Point pt0)//косинус угла pt1-pt0-pt2
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

static void findSquares(const Mat& image, vector<vector<Point>>& squares)//Возвращает последовательность квадратов, обнаруженных на изображении.
{
    squares.clear();
    Mat pyr, timg, gray, gray0(image.size(), CV_8U);

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point>> contours;
    vector<cv::Vec4i> hierarchy;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for(uint8_t l = 0; l < N; l++)
        {
        // hack: use Canny instead of zero threshold level.
        // Canny helps to catch squares with gradient shading
            if(l == 0)
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, UMat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l + 1) * 255 / N;
            }
            // find contours and store them all as a list
            findContours(gray, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
            vector<Point> approx;
            // test each contour
            for(size_t i = 0; i < contours.size(); i++)
            {
            // approximate contour with accuracy proportional
            // to the contour perimeter
                approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);
                // square contours should have 4 vertices after approximation
                    // relatively large area (to filter out noisy contours)
                    // and be convex.
                    // Note: absolute value of an area is used because
                    // area may be positive or negative - in accordance with the
                    // contour orientation
                if(approx.size() == 4 &&
                   fabs(contourArea(approx)) > 1000 &&
                   isContourConvex(approx))
                {
                    double maxCosine = 0;
                    for(uint8_t j = 2; j < 5; j++)
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                        cout << maxCosine << endl;
                    }
                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                        // vertices to resultant sequence
                    if( maxCosine < 0.3 ) squares.push_back(approx);//cos(80) = 0.1736481777 cos(100) = -0.1736481777
                }
            }
        }
    }
}

// the function draws all the squares in the image
static void drawSquares(Mat& _image, const vector<vector<Point> >& squares)
{
    UMat image = _image.getUMat(ACCESS_WRITE);
    for(size_t i = 0; i < squares.size(); i++)
    {
        const Point* p = &squares[i][0];
        int n = static_cast<int>(squares[i].size());
        polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 1, LINE_AA);
    }
}

// draw both pure-C++ and ocl square results onto a single image
static Mat drawSquaresBoth( const Mat& image,
                            const vector<vector<Point>>& sqs)
{
    Mat imgToShow(Size(image.cols, image.rows), image.type());
    image.copyTo(imgToShow);
    drawSquares(imgToShow, sqs);
    return imgToShow;
}

int main( int argc, char** argv )
{
    uint8_t iterations = 10;

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        printHelp();
    }
    String pathFile = parser.get<String>("path");
    pathFile += parser.get<String>("@file");
    vector<vector<Point>> squares;
    Mat image = imread(pathFile, IMREAD_COLOR);
    if(image.empty())
    {
        cout << "Couldn't load " << pathFile << endl;
        parser.printMessage();
        return EXIT_FAILURE;
    }
    uint8_t j = iterations;
    int64 t_cpp = 0;
    //warm-ups
    cout << "warming up ..." << endl;
    findSquares(image, squares);
    do
    {
        int64 t_start = getTickCount();
        findSquares(image, squares);
        t_cpp += getTickCount() - t_start;
        t_start  = getTickCount();
        cout << "run loop: " << j << endl;
    }
    while(--j);
    cout << "average time: " << 1000.0f * static_cast<double>(t_cpp)
            / getTickFrequency() / iterations << "ms" << endl;
    Mat result = drawSquaresBoth(image, squares);
    imshow(wndname, result);
    waitKey(3000);
    return EXIT_SUCCESS;
}
