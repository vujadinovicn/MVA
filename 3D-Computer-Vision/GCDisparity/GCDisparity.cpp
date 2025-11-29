// Imagine++ project
// Project: GraphCutsDisparity
// Author: Renaud Marlet/Pascal Monasse
// Student: Nemanja Vujadinovic

#include <Imagine/Images.h>
#include <iostream>
#include <algorithm>
#include <string>
#include "maxflow/graph.h"
using namespace Imagine;
using namespace std;

typedef Image<Imagine::byte> byteImage;
typedef Image<double> doubleImage;

// Default data
// const char *DEF_im1=srcPath("face0.png"), *DEF_im2=srcPath("face1.png");
std::string DEF_im1=srcPath("face0.png"), DEF_im2=srcPath("face1.png");
// std::string DEF_im1 = srcPath("face0.png");
// std::string DEF_im2 = srcPath("face1.png");
static int dmin=10, dmax=55; // Min and max disparities

// Parameters of the algorithm
// OPTIMIZATION: to make the program faster, a zoom factor is used to
// down-sample the input images on the fly. You will
// only look at pixels (win+zoom*i,win+zoom*j) with win the radius of patch.
const int win = (7-1)/2;    // Correlation patches of size (2n+1)*(2n+1)
const float lambdaf = 0.1;  // Weight of regularization (smoothing) term
const int zoom = 2;         // Zoom factor (to speedup computations)
const float sigma = 6/zoom; // Gaussian blur parameter for disparity
// Energy discretization precision (as we build a graph with 'int' weights)
const int wcc = std::max(1+int(1/lambdaf),20);
const int lambda = lambdaf*wcc; // Regularization term (must be >= 1)

// Neighbor offsets (right and down)
// Even though we use 4-connectivity, we only need to create edges
// in two directions (right and down) to avoid duplicating edges
// since we are going right-down with pixels when creating the graph.
static const int dx[]={+1,  0};
static const int dy[]={ 0, +1};
static const int noOfNeighbors = (sizeof(dx)/sizeof(*dx));

Image<Color> displayDisp(const Image<int>& disp) {
    Image<Color> im(disp.width(), disp.height());
    for(int j=0; j<disp.height(); j++)
        for(int i=0; i<disp.width(); i++) {
            if(disp(i,j)<dmin || disp(i,j)>dmax)
                im(i,j) = CYAN;
            else {
                int g = 255*(disp(i,j)-dmin)/(dmax-dmin);
                im(i,j)= Color(g,g,g);
            }
        }
    return im;
}

#ifdef IMAGINE_OPENGL
void doc() {
    cout << "***** 3D mesh renderings *****" << endl;
    cout << "- Button 1: toggle textured or gray rendering" << endl;
    cout << "- SHIFT+Button 1: rotate" << endl;
    cout << "- SHIFT+Button 3: translate" << endl;
    cout << "- Mouse wheel: zoom" << endl;
    cout << "- SHIFT+a: zoom out" << endl;
    cout << "- SHIFT+z: zoom in" << endl;
    cout << "- SHIFT+r: recenter camera" << endl;
    cout << "- SHIFT+m: toggle solid/wire/points mode" << endl;
    cout << "- Button 3: exit" << endl;
}

// Display 3D mesh renderings
void control3D(const Mesh& Mt, const Mesh& Mg) {
    setActiveWindow( openWindow3D(512, 512, "3D") );
    showMesh(Mt);
    bool textured = true;
    while (true) {
        Event evt;
        getEvent(5, evt);
        // On mouse button 1
        if (evt.type == EVT_BUT_ON && evt.button == 1) {
            // Toggle textured rendering and gray rendering
            if (textured) {
                hideMesh(Mt,false);
                showMesh(Mg,false);
            } else {
                hideMesh(Mg,false);
                showMesh(Mt,false);
            }
            textured = !textured;
        }
        // On mouse button 3
        if (evt.type == EVT_BUT_ON && evt.button == 3)
            break;
    }
}
#endif

void show3D(const byteImage& I, const doubleImage& D, int zoom) {
#ifdef IMAGINE_OPENGL
    cout << "Click to compute depth map and 3D mesh renderings... " << flush;
    click();

    // Compute 3D point cloud: magic constants depending on camera pose
    const float f  = 750;  // Focal
    const float d0 = 100;  // Disparity for infinite depth (due to crop)
    const float B  = -0.20;// Baseline

    const int nx=D.width(), ny=D.height();
    FMatrix<float,3,3> K(0.0f);
    K(0,0)= -f/zoom; K(0,2)=nx/2;
    K(1,1)=  f/zoom; K(1,2)=ny/2;
    K(2,2)=1.0f;
    K = inverse(K);
    K /= K(2,2);
    Array<FloatPoint3> p(nx*ny);
    Array<Color> pcol(nx*ny);
    for(int j=0; j<ny; j++)
        for(int i=0; i<nx; i++) {
            float z = f*B/(d0+D(i,j));
            FloatPoint3 pt((float)i,(float)j,1.0f);
            p[i+nx*j] = K*pt*z;
            pcol[i+nx*j] = Color(I(i*zoom,j*zoom));
        }
    // Create mesh from 3D point cloud
    Array<Triangle> t(2*(nx-1)*(ny-1));
    Array<Color> tcol(2*(nx-1)*(ny-1));
    for(int j=0; j+1<ny; j++)
        for(int i=0; i+1<nx; i++) {
            // Create triangles with next pixels in line/column
            t[2*(i+j*(nx-1))]   = Triangle(i+nx*j,   i+1+nx*j,     i+nx*(j+1));
            t[2*(i+j*(nx-1))+1] = Triangle(i+1+nx*j, i+1+nx*(j+1), i+nx*(j+1));
            tcol[2*(i+j*(nx-1))]   = pcol[i+nx*j];
            tcol[2*(i+j*(nx-1))+1] = pcol[i+nx*j];
        }
    // Mesh with texture from original image
    Mesh Mt(p.data(), nx*ny, t.data(), 2*(nx-1)*(ny-1), 0, 0, FACE_COLOR);
    Mt.setColors(TRIANGLE, tcol.data());
    // Mesh with artificial light
    Mesh Mg(p.data(), nx*ny, t.data(), 2*(nx-1)*(ny-1), 0, 0,
            CONSTANT_COLOR, SMOOTH_SHADING);
    cout << "done" << endl;

    doc();
    control3D(Mt, Mg);
#else
    cout << "No 3D: Imagine++ not built with OpenGL support" << endl;
#endif
}

// Return image of mean intensity value over (2win+1)x(2win+1) patch
doubleImage meanImage(const byteImage& I) {
    int w = I.width(), h = I.height();
    doubleImage IM(w,h);
    double area = (2*win+1)*(2*win+1);
    for(int j=0; j<h; j++)
        for(int i=0; i<w; i++) {
            if (j-win<0 || j+win>=h || i-win<0 ||i+win>=w) {
                IM(i,j)=0;
                continue;
            }
            double sum = 0;
            for(int y=-win; y<=win; y++)
                for(int x=-win; x<=win; x++)
                    sum += I(x+i,y+j);
            IM(i,j) = sum / area;
        }
    return IM;
}

// Compute correlation between two pixels in images 1 and 2
double correl(const byteImage& I1,    // Image 1
              const doubleImage& I1M, // Image of mean value over patch
              const byteImage& I2,    // Image2
              const doubleImage& I2M, // Image of mean value over patch
              int u1, int v1,         // Pixel of interest in image 1
              int u2, int v2) {       // Pixel of interest in image 2
    double c=0;
    for(int y=-win; y<=win; y++)
        for(int x=-win; x<=win; x++) 
            c += (I1(u1+x,v1+y)-I1M(u1,v1)) * (I2(u2+x,v2+y)-I2M(u2,v2));
    return c / ((2*win+1)*(2*win+1));
}

// Compute ZNCC between two patches in images 1 and 2
double zncc(const byteImage& I1,    // Image 1
            const doubleImage& I1M, // Image of mean intensity value over patch
            const byteImage& I2,    // Image2
            const doubleImage& I2M, // Image of mean intensity value over patch
            int u1, int v1,         // Pixel of interest in image 1
            int u2, int v2) {       // Pixel of interest in image 2
    double var1 = correl(I1, I1M, I1, I1M, u1, v1, u1, v1);
    if(var1 == 0)
        return 0;
    double var2 = correl(I2, I2M, I2, I2M, u2, v2, u2, v2);
    if(var2 == 0)
        return 0;
    return correl(I1, I1M, I2, I2M, u1, v1, u2, v2) / sqrt(var1 * var2);
}

int getWeightFromZncc(double znccScore) {
    // clamp zncc to [-1.0, 1.0] to avoid small float numerical errors
    if (znccScore > 1.0) znccScore = 1.0;
    if (znccScore < -1.0) znccScore = -1.0;
    double rho = (znccScore < 0.0) ? 1.0 : sqrt(1.0 - znccScore);
    int weight = int(std::round(wcc * rho));
    return weight;
}

/// Create graph
/// The graph library works with node numbers. To clarify the setting, create
/// a formula to associate a unique node number to a triplet (x,y,d) of pixel
/// coordinates and disparity.
/// The library assumes an edge consists of a pair of oriented edges, one in
/// each direction. Put correct weights to the edges, such as 0, INF, or
/// an intermediate weight.
void build_graph(Graph<int,int,int>& G,
                 const byteImage& I1, const byteImage& I2,
                 int nx, int ny, int nd) {
    const int INF=1000000; // "Infinite" value for edge impossible to cut
    // Precompute images of mean intensity value over patch
    doubleImage I1M = meanImage(I1), I2M = meanImage(I2);

    // ------------- TODO -------------

    int numberOfNodes = nx * ny * nd; // there are nx*ny pixels and nd disparity levels for each pixel
    G.add_node(numberOfNodes);

    // formula to associate a unique node number
    // find pixel in image grid with y + ny * x
    // multiply by nd to account for disparity layers and add d for current layer
    auto nbNode = [ny, nd](int x, int y, int d){ 
        return d + nd * (y + ny * x);
    };

    // precompute weights by computing wcc * rho(zncc) for every node
    std::vector<int> weights(nx * ny * nd, wcc); // initialize all weights to wcc

    for (int x = 0; x < nx; x++){
        int u = win + zoom*x; // x coordinate in original I1 image
        for (int y = 0; y < ny; y++){
            int v = win + zoom*y; // y coordinate in original I1 image
            
            if (u - win < 0 || u + win >= I1.width() || v - win < 0 || v + win >= I1.height()) // skip if windows are not fully in image; weight is already initialized to wcc
                continue;

            for (int l = 0; l < nd; l++){ // iterate through layers (disparity range)
                int d = dmin + l;
                int u2 = u + d; // corresponding x coordinate in original I2 image
                if (u2 - win < 0 || u2 + win >= I2.width() || v - win < 0 || v + win >= I2.height()) // skip if windows are not fully in image; weight is already initialized to wcc
                    continue;

                // compute data term
                double znccScore = zncc(I1, I1M, I2, I2M, u, v, u2, v);
                int weight = getWeightFromZncc(znccScore);
                int node = nbNode(x, y, l);
                weights[node] = weight; // weight for edge t_{j}^p = D_{p}j
            }
        }
    }

    for (int x = 0; x < nx; x++){
        for (int y=0; y < ny; y++){
            // compute penalty K for current node/pixel
            int nbValidNeighbors = (x > 0) + (x + 1 < nx) + (y > 0) + (y + 1 < ny); // get number of valid neighbors
            int K = 1 + (nd - 1) * lambda * nbValidNeighbors;

            for (int l=0; l < nd; l++){
                int currentNode = nbNode(x, y, l); // take current layer node for (x, y) pixel
                int currentWeight = weights[currentNode] + K; // current node weight with K penalty

                // create t-link between s and first layer for each pixel
                if (l == 0){
                    G.add_tweights(currentNode, currentWeight, 0);
                }
                //
                else {
                    int previousLayerNode = nbNode(x, y, l-1); // node in previous layer (l-1)
                    // connect previous and current layer with current layer weight in the forward direction
                    // set weight to INF in backward direction to enforce one monotone cut along the column
                    // based on ref: https://ieeexplore.ieee.org/document/7299150
                    G.add_edge(previousLayerNode, currentNode, currentWeight, INF);
                }
                // create t-link between last layer and t for each pixel
                if (l == nd - 1){
                    G.add_tweights(currentNode, 0, currentWeight);
                }

                // create edges between neighboring pixels for smoothness term
                for (int n = 0; n < noOfNeighbors; n++){
                    int xn = x + dx[n]; int yn = y + dy[n];
                    if (xn >= 0 && xn < nx && yn >= 0 && yn < ny){ // check if neighbor is in image bounds
                        int neighborNode = nbNode(xn, yn, l);
                        G.add_edge(currentNode, neighborNode, lambda, lambda);
                    }
                }
            }
        }
    }
}

/// Extract disparity from minimum cut.
/// w2 is the width of image2, used to detect pixels with no valid patch.
doubleImage decode_graph(Graph<int,int,int>& G, int nx, int ny, int nd, int w2){
    doubleImage D(nx,ny);
    // ------------- TODO -------------
    // The following is dummy code, replace by your own
    auto nbNode = [ny, nd](int x, int y, int d){ 
        return d + nd * (y + ny * x);
    };

    for(int x = 0; x < nx; x++) {
        for(int y = 0; y < ny; y++) {
            int d = 0; // disparity level/layer

            // while current disparity level is on SOURCE side and lower than maximum layer, increase disparity
            while (d < nd && G.what_segment(nbNode(x, y, d)) == Graph<int,int,int>::SOURCE){
                d++;
            }
            
            D(x,y) = dmin + d; // assign disparity value
        }
    }
    return D;
}
    
// Load two rectified images.
// Compute the disparity of image 2 w.r.t. image 1.
// Display disparity map.
// Display 3D mesh of corresponding depth map.
int main(int argc, char* argv[]) {
    if(argc!=1 && argc!=5) {
        cerr << "Usage: " << argv[0] << " im1 im2 dmin dmax" << endl;
        return 1;
    }
    // const char *im1=DEF_im1, *im2=DEF_im2;
    std::string im1=DEF_im1, im2=DEF_im2;
    if(argc>1) {
        im1 = argv[1]; im2=argv[2]; dmin=stoi(argv[3]); dmax=stoi(argv[4]);
    }
    cout << "Loading images... " << flush;
    byteImage I1,I2;
    if(!load(I1, im1) || !load(I2,im2)) {
        cerr << "Error loading image files" << endl;
        return 1;
    }
    cout << "done" << endl;

    cout << "Parameters: " << "d=" << dmin << "..." << dmax
         << ", win="<<win << ", lambda="<<lambdaf << ", sigma="<<sigma
         << ", zoom="<<zoom << endl;

    cout << "Displaying images... " << flush;
    int w1=I1.width(), w2=I2.width(), h=I1.height();
    openWindow(w1+w2, h);
    display(I1); display(I2,w1,0);
    cout << "done" << endl;

    // Zoomed image dim, disregarding borders (strips of width the patch radius)
    const int nx=(w1-2*win)/zoom, ny=(h-2*win)/zoom;
    const int nd=dmax-dmin; // Disparity range

    cout << "Constructing graph (be patient)... " << flush;
    Graph<int,int,int> G(nx*ny*nd,10*nx*ny*nd);
    build_graph(G, I1, I2, nx, ny, nd);
    cout << "done" << endl;

    cout << "Computing minimum cut... " << flush;
    int f = G.maxflow();
    cout << "done" << endl << "  max flow = " << f << endl;

    cout << "Extracting disparity map from minimum cut... " << flush;
    doubleImage D=decode_graph(G, nx, ny, nd, I2.width());
    cout << "done" << endl;

    cout << "Displaying disparity map... " << flush;
    fillRect(0,0,w1,h,CYAN);
    display(enlarge(displayDisp(D),zoom),win,win);
    cout << "done" << endl;
    cout << "Click to compute and display blurred disparity map... " << flush;
    click();
    D=blur(D,sigma);
    display(enlarge(displayDisp(D),zoom),win,win);
    cout << "done" << endl;

    show3D(I1.getSubImage(win,win,w1-2*win,h-2*win), D, zoom);
    endGraphics();
    return 0;
}
