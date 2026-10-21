// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Student: Nemanja Vujadinovic

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;

// Record clicks in two images, until right button click
void getClicks(Window w1, Window w2,
               vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {
    // ------------- TODO/A completer ----------
    pts1.clear(); pts2.clear();
    Window currentWindow; // tracks which window did user click on
    int button, subWindow;

    cout << "Click on images w1 and w2. Click at least 4 points on both. Click right click to stop" << endl;

    while (true) {
        IntPoint2 newPoint;
        button = anyGetMouse(newPoint, currentWindow, subWindow);
        if (button == 3) // stop if right button clicked
            break;
        
        setActiveWindow(currentWindow);
        drawCircle(newPoint, 10, RED, 5);

        if (currentWindow == w2)
            pts2.push_back(newPoint);
        else if (currentWindow == w1)
            pts1.push_back(newPoint);
    }

    if (pts1.size() != pts2.size()) {
        cout << "Warning: Point count mismatch! This can lead to inaccurate results due to missing correspondences." << endl;
    }

}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    Vector<double> B(2*n);
    // ------------- TODO/A completer ----------

    // Build matrices A and B for the linear system Ah = B.
    // For each corresponding points (pts1[i], pts2[i]), there are two equations:
    //   x' = (h11*x + h12*y + h13) / (h31*x + h32*y + 1)
    //   y' = (h21*x + h22*y + h23) / (h31*x + h32*y + 1)
    // For each index, fill two rows in A and B:
    for (size_t i = 0; i < n; i++) {
        // coordinates of points in I1 (first image)
        double x = pts1[i].x();
        double y = pts1[i].y();

        // coordinates of points in I2 (second image)
        double xPrime = pts2[i].x();
        double yPrime = pts2[i].y();

        // even row for x' equation
        A(2*i,0) = x;      A(2*i,1) = y;      A(2*i,2) = 1;
        A(2*i,3) = 0;      A(2*i,4) = 0;      A(2*i,5) = 0;
        A(2*i,6) = -xPrime*x; A(2*i,7) = -xPrime*y;
        B[2*i] = xPrime;

        // odd row for y' equation
        A(2*i+1,0) = 0;    A(2*i+1,1) = 0;    A(2*i+1,2) = 0;
        A(2*i+1,3) = x;    A(2*i+1,4) = y;    A(2*i+1,5) = 1;
        A(2*i+1,6) = -yPrime*x; A(2*i+1,7) = -yPrime*y;
        B[2*i+1] = yPrime;
    }

    B = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    // Sanity check
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;    
}

// Convert a 3d float vector to Color and clamp the values to [0,255]
Color vectorToColor(const FVector<float,3>& v) {
    auto clampAndCast = [](float a)->Imagine::byte {
        if (a < 0.f)   a = 0.f;
        if (a > 255.f) a = 255.f;
        return static_cast<Imagine::byte>(a + 0.5f);
    };
    return Color(clampAndCast(v[0]), clampAndCast(v[1]), clampAndCast(v[2]));
}

// Compute the average color of two colors
Color averageColors(const Color& c1, const Color& c2) {
    return Color((int(c1.r()) + int(c2.r())) / 2,
                 (int(c1.g()) + int(c2.g())) / 2,
                 (int(c1.b()) + int(c2.b())) / 2);
}

// Panorama construction
void panorama(const Image<Color,2>& I1, const Image<Color,2>& I2,
              Matrix<float> H) {
    Vector<float> v(3);
    float x0=0, y0=0, x1=I2.width(), y1=I2.height();
    for(int i=0; i<2; i++)
        for(int j=0; j<2; j++) {
            v[0] = j*I1.width(); v[1] = i*I1.height(); v[2] = 1;
            v=H*v; v/=v[2];
            growTo(x0, y0, x1, y1, v[0], v[1]);
        }
    cout << "Rectangle of mosaic in I2 coordinates:" << endl;
    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1 << endl;

    Image<Color> I(int(x1-x0), int(y1-y0));
    setActiveWindow( openWindow(I.width(), I.height(), "Panorama") );
    I.fill(WHITE);
    // ------------- TODO/A completer ----------

    Matrix<float> H_inv = inverse(H); // Compute inverse of H since we are pulling pixels from I1 to I
    
    // Iterate through each pixel in the panorama
    for (int y = 0; y < I.height(); y++) {
        for (int x = 0; x < I.width(); x++) {
            bool isInpaintedFromI2 = false; // Flag to check if pixel was inpainted from I2
            
            Vector<float> x1Point(3), x2Point(3); // Homogeneous coordinates in I1 and I2
            x2Point[0] = x + x0; x2Point[1] = y + y0; x2Point[2] = 1; // Assign panorama pixel (x, y) to I2 coordinates (x+x0, y+y0)

            // Check if the current panorama pixel maps inside I2
            if (0 <= x2Point[0] && x2Point[0] < I2.width() &&
                0 <= x2Point[1] && x2Point[1] < I2.height()) {
                // Copy pixel from I2 to panorama
                I(x, y) = I2(x2Point[0], x2Point[1]);
                isInpaintedFromI2 = true;
            }

            x1Point = H_inv * x2Point; // Map I2 pixel to I1 using inverse homography
            x1Point /= x1Point[2]; // Normalize to inhomogeneous coordinates 

            // Check if the mapped pixel is inside I1 and interpolate color
            if (0 <= x1Point[0] && x1Point[0] < I1.width() &&
                0 <= x1Point[1] && x1Point[1] < I1.height()) 
            {
                FVector<float,3> interpolatedVector = I1.interpolate(x1Point[0], x1Point[1]);
                Color interpolatedColor = vectorToColor(interpolatedVector);

                // If pixel is already inpainted from I2, average the colors
                if (isInpaintedFromI2) {
                    Color panoramaColor = I(x, y);
                    I(x, y) = averageColors(interpolatedColor, panoramaColor);
                } else {
                    I(x, y) = interpolatedColor;
                }
            }
        }
    }
    display(I,0,0);

}

// Main function
int main(int argc, char* argv[]) {
    string s1 = argc>2? argv[1]: srcPath("image0006.jpg");
    string s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    getClicks(w1, w2, pts1, pts2);

    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
