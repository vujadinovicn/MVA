// Imagine++ project
// Project:  Seeds
// Author:   Pascal Monasse
// Student: Nemanja Vujadinovic

#include <Imagine/Images.h>
#include <queue>
#include <string>
#include <iostream>
#include <chrono> // at top

using namespace Imagine;
using namespace std;

// Default data
const string DEF_im1=srcPath("dataset/Dolls/view1.png"), DEF_im2=srcPath("dataset/Dolls/view5.png");
static int dmin=-85, dmax=0; // Min and max disparities

/// Min NCC for a seed
static const float nccSeed=0.95f;

/// Radius of patch for correlation
static const int win=(7-1)/2;
/// To avoid division by 0 for constant patch
static const float EPS=0.1f;

/// A seed
struct Seed {
    Seed(int x0, int y0, int d0, float ncc0)
    : x(x0), y(y0), d(d0), ncc(ncc0) {}
    int x,y, d;
    float ncc;
};

/// Order by NCC
bool operator<(const Seed& s1, const Seed& s2) {
    return (s1.ncc<s2.ncc);
}

/// 4-neighbors
static const int dx[]={+1,  0, -1,  0};
static const int dy[]={ 0, -1,  0, +1};

/// Display disparity map
static Image<Color> displayDisp(const Image<int>& disp, Window W, int subW) {
    Image<Color> im(disp.width(), disp.height());
    Image<short> disp16(disp.width(), disp.height());
    for(int j=0; j<disp.height(); j++)
        for(int i=0; i<disp.width(); i++) {
            if(disp(i,j)<dmin || disp(i,j)>dmax)
                im(i,j) = Color(0,0,0);
            else {
                // int g = 255*(disp(i,j)-dmin)/(dmax-dmin);
                int g = 255*(dmax-disp(i,j))/(dmax-dmin);
                im(i,j)= Color(g,g,g);
            }
        }
    setActiveWindow(W,subW);
    display(im);
    showWindow(W,subW);
    return im;
}

// Map disparity -> grayscale image (no GUI)
static Image<Color> dispToImage(const Image<int>& disp, const string& currentDir) {
    Image<Color> im(disp.width(), disp.height());
    Image<octet> disp8(disp.width(), disp.height());
    const int offset = (dmin < 0) ? -dmin : 0;   // e.g., 85 if range is [-85..0], else 0

    for (int y = 0; y < disp.height(); ++y) {
        for (int x = 0; x < disp.width(); ++x) {
            int v = disp(x,y);

            if (v < dmin || v > dmax) {
                disp8(x,y) = (octet)0;
            } else {
                int raw = v * (-1)+1;                    // now in [0..85] if range was [-85..0]
                if (raw < 0) raw = 0;
                if (raw > 255) raw = 255;
                disp8(x,y) = (octet)raw;
            }

            if (v < dmin || v > dmax) {
                im(x,y) = Color(0,0,0);
            } else {
                int g = 255*(dmax - v)/(dmax - dmin);
                im(x,y) = Color(g,g,g);
            }
        }
    }

    save(disp8, srcPath(currentDir + "/disp_region_raw.png"));
    return im;
}

/// Show 3D window
static void show3D(const Image<Color>& im, const Image<int>& disp) {
#ifdef IMAGINE_OPENGL // Imagine++ must have been built with OpenGL support...
    // Intrinsic parameters given by Middlebury website
    std::cerr << "[show3D] IMAGINE_OPENGL is defined\n";
    const float f=3740;
    const float d0=-200; // Doll images cropped by this amount
    const float zoom=2; // Half-size images, should double measured disparity
    const float B=0.160; // Baseline in m
    FMatrix<float,3,3> K(0.0f);
    K(0,0)=-f/zoom; K(0,2)=disp.width()/2;
    K(1,1)= f/zoom; K(1,2)=disp.height()/2;
    K(2,2)=1.0f;
    K = inverse(K);
    K /= K(2,2);
    std::vector<FloatPoint3> pts;
    std::vector<Color> col;
    for(int j=0; j<disp.height(); j++)
        for(int i=0; i<disp.width(); i++)
            if(dmin<=disp(i,j) && disp(i,j)<=dmax) {
                float z = B*f/(zoom*disp(i,j)+d0);
                FloatPoint3 pt((float)i,(float)j,1.0f);
                pts.push_back(K*pt*z);
                col.push_back(im(i,j));
            }
    if(pts.empty()) {
        std::cerr << "No 3D point..." << std::endl;
        return;
    }
    Mesh mesh(&pts[0], pts.size(), 0,0,0,0,VERTEX_COLOR);
    mesh.setColors(VERTEX, &col[0]);
    Window W = openWindow3D(512,512,"3D");
    setActiveWindow(W);
    showMesh(mesh);
#else
    std::cerr << "[show3D] IMAGINE_OPENGL is not defined\n";
#endif
}

/// Correlation between patches centered on (i1,j1) and (i2,j2). The values
/// m1 or m2 are subtracted from each pixel value.
static float correl(const Image<octet>& im1, int i1,int j1,float m1,
                    const Image<octet>& im2, int i2,int j2,float m2) {
    // ------------- TODO -------------
    // for each x1€{i1-win, i1+win}, y1€{j1-win, j1+win} and x2€{i2-win, i2+win}, y2€{j2-win, j2+win} we compute num, denum1, denum2
    float num  = 0.0f; // sum((im1(x1,y1) - m1) * (im2(x2,y2) - m2))
    float denum1 = 0.0f; // sum((im1(x1, y1) - m1)^2)
    float denum2 = 0.0f; // sum((im2(x2, y2) - m2)^2)

    for (int dy = -win; dy <= win; dy++){
        for (int dx = -win; dx <= win; dx++){
            float mn1 = float(im1(i1 + dx, j1 + dy)) - m1; // mean-normalized pixel from im1
            float mn2 = float(im2(i2 + dx, j2 + dy)) - m2; // mean-normalized pixel from im2

            num  += mn1 * mn2;
            denum1 += mn1 * mn1;
            denum2 += mn2 * mn2;
        }
    }

    float dist = num / (sqrt(denum1 * denum2) + EPS);
    return dist;
}

/// Sum of pixel values in patch centered on (i,j).
static float sum(const Image<octet>& im, int i, int j) {
    float s=0.0f;
    // ------------- TODO -------------
    for (int dy = -win; dy <= win; dy++){
        for (int dx = -win; dx <= win; dx++){
            s += im(i+dx, j+dy);
        }
    }
    return s;
}

/// Centered correlation of patches of size 2*win+1.
static float ccorrel(const Image<octet>& im1,int i1,int j1,
                     const Image<octet>& im2,int i2,int j2) {
    float m1 = sum(im1,i1,j1);
    float m2 = sum(im2,i2,j2);
    int w = 2*win+1;
    return correl(im1,i1,j1,m1/(w*w), im2,i2,j2,m2/(w*w));
}

/// Compute disparity map from im1 to im2, but only at points where NCC is
/// above nccSeed. Set to true the seeds and put them in Q.
static void find_seeds(Image<octet> im1, Image<octet> im2,
                       float nccSeed,
                       Image<int>& disp, Image<bool>& seeds,
                       std::priority_queue<Seed>& Q) {
    disp.fill(dmin-1);
    seeds.fill(false);
    while(! Q.empty())
        Q.pop();

    const int maxy = std::min(im1.height(),im2.height());
    const int refreshStep = (maxy-2*win)*5/100;
    for(int y=win; y+win<maxy; y++) {
        if((y-win-1)/refreshStep != (y-win)/refreshStep)
            std::cout << "Seeds: " << 5*(y-win)/refreshStep <<"%\r"<<std::flush;
        for(int x=win; x+win<im1.width(); x++) {
            // ------------- TODO -------------
            // Hint: just ignore windows that are not fully in image
            // if dense pass, only compute disparity map from im1 to im2 by highest NCC score
            // if seed pass, compute best NCC and disparity, and if above threshold, put seed in queue

            int bestD; // disparity with highest NCC for (x,y)
            float bestNcc= -1e8; // highest NCC for (x,y)
            bool updated = false; // flag to check if we updated bestNcc

            for (int d = dmin; d <= dmax; d++){ // iterate through disparity range
                int x2 = x + d; // x position in im2 for disparity d
                if (x2 - win < 0 || x2 + win >= im2.width()) // check and skip if x2's window is not fully in image
                    continue;

                float ncc = ccorrel(im1, x, y, im2, x2, y);
                
                if (ncc > bestNcc){ // update highest NCC and corresponding disparity
                    bestD = d;
                    bestNcc = ncc;
                    updated = true;
                }
            }
            
            if (!updated) // no valid NCC found
                continue;
            
            // check if find_seeds() is actually dense pass (in main(), nccSeed is set to -1.0f for dense pass) 
            // if it is dense pass, only compute disparity map and skip enqueing seeds
            if (nccSeed == -1.0f) {
                disp(x, y) = bestD;
            }
            else if (bestNcc >= nccSeed) { // if find_seeds() is seed pass and bestNcc is above threshold, put seed in queue
                disp(x, y) = bestD; // update disparity
                Q.push(Seed(x, y, bestD, bestNcc)); // create and add new seed to queue
                seeds(x, y) = true; // mark as seed
            }
        }
    }
    std::cout << std::endl;
}

/// Propagate seeds
static void propagate(Image<octet> im1, Image<octet> im2,
                      Image<int>& disp, Image<bool>& seeds,
                      std::priority_queue<Seed>& Q) {
    const int maxy = std::min(im1.height(),im2.height());
    while(! Q.empty()) {
        Seed s=Q.top();
        Q.pop();
        for(int i=0; i<4; i++) {
            int x=s.x+dx[i], y=s.y+dy[i];
            if(0<=x-win && x+win<im1.width() && 0<=y-win && y+win<maxy &&
               ! seeds(x,y)) {
                // ------------- TODO -------------
                // set disparity of single 4-neighbour of the seed by highest NCC score among disparities s.d-1, s.d, s.d+1

                int bestD; // disparity with highest NCC for (x,y)
                float bestNcc= -1e8; // highest NCC for (x,y)
                bool updated = false; // flag to check if we updated bestNcc

                for (int d = s.d-1; d <= s.d+1; d++) { // iterate through disparities around seed's disparity
                    int x2 = x + d; // x position in im2 for disparity d
                    // check and skip if x2's window is not fully in image
                    // check and skip if disparity is out of range
                    if (x2 - win < 0 || x2 + win >= im2.width() || d < dmin || d > dmax)
                        continue;

                    float ncc = ccorrel(im1, x, y, im2, x2, y);
                    
                    if (ncc > bestNcc) { // update highest NCC and corresponding disparity
                        bestD = d;
                        bestNcc = ncc;
                        updated = true;
                    }
                }
                
                if (updated) { // if we found a valid NCC
                    disp(x, y) = bestD; // update disparity
                    Q.push(Seed(x, y, bestD, bestNcc)); // create and add new seed to queue
                    seeds(x, y) = true; // mark as seed
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if(argc!=1 && argc!=5) {
        cerr << "Usage: " << argv[0] << " im1 im2 dmin dmax" << endl;
        return 1;
    }
    for (int i = 0; i < 6; i++) {
        string im1, im2, currentDir;
        if (i == 0) { im1=srcPath("dataset/Art/view1.png"), im2=srcPath("dataset/Art/view5.png"); currentDir = "dataset/Art"; }
        else if (i == 1) { im1=srcPath("dataset/Books/view1.png"), im2=srcPath("dataset/Books/view5.png"); currentDir = "dataset/Books/run2"; }
        else if (i == 2) { im1=srcPath("dataset/Dolls/view1.png"), im2=srcPath("dataset/Dolls/view5.png"); currentDir = "dataset/Dolls/run2"; }
        else if (i == 3) { im1=srcPath("dataset/Laundry/view1.png"), im2=srcPath("dataset/Laundry/view5.png"); currentDir = "dataset/Laundry/run2"; }
        else if (i == 4) { im1=srcPath("dataset/Moebius/view1.png"), im2=srcPath("dataset/Moebius/view5.png"); currentDir = "dataset/Moebius/run2"; }
        else if (i == 5) { im1 =srcPath("dataset/Reindeer/view1.png"), im2=srcPath("dataset/Reindeer/view5.png"); currentDir = "dataset/Reindeer/run2"; }
        if(argc>1) { im1=argv[1]; im2=argv[2]; dmin=stoi(argv[3]); dmax=stoi(argv[4]); }

        // Load
        Image<Color> I1, I2;
        if(!load(I1,im1) || !load(I2,im2)) { cerr<< "Error loading image files\n"; return 1; }

        Image<int> disp(I1.width(), I1.height());
        Image<bool> seeds(I1.width(), I1.height());
        std::priority_queue<Seed> Q;

        using clk = std::chrono::high_resolution_clock;

        auto t0 = clk::now();
        find_seeds(I1, I2, -1.0f, disp, seeds, Q);               // dense pass
        auto t1 = clk::now();
        find_seeds(I1, I2, nccSeed, disp, seeds, Q);             // seeds only
        auto t2 = clk::now();
        propagate(I1, I2, disp, seeds, Q);                       // propagation
        auto t3 = clk::now();

        // Save intermediate & final visualizations (headless)
        save(dispToImage(disp, currentDir), srcPath(currentDir + "/disp_region_vis.png"));

        auto ms_dense = std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count();
        auto ms_seed  = std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count();
        auto ms_prop  = std::chrono::duration_cast<std::chrono::seconds>(t3-t2).count();
        auto ms_total = std::chrono::duration_cast<std::chrono::seconds>(t3-t0).count();

        cout << "Timing (ms): dense=" << ms_dense
            << "  seeds=" << ms_seed
            << "  propagate=" << ms_prop
            << "  total=" << ms_total << "\n";
    }
    return 0;
}