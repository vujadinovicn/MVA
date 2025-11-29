// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure
static const float NORMALIZATION_FACTOR = 0.001f; // Constant for normalization of points for 8-point algorithm

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// Return 9x9 A matrix for 8-point algorithm (x^TFx'=0 => Af=0)
FMatrix<float,9,9> getA(const vector<Match>& matches){
    FMatrix<float,9,9> A;
    for (int i = 0; i < 8; i++){
        // Obtain and normalize coordinates to be of order 1
        // F will be denormalized later by the same factors
        float x1 = matches[i].x1 * NORMALIZATION_FACTOR;
        float y1 = matches[i].y1 * NORMALIZATION_FACTOR;
        float x2 = matches[i].x2 * NORMALIZATION_FACTOR;
        float y2 = matches[i].y2 * NORMALIZATION_FACTOR;

        // Fill row i of A
        // For each match (x1, y1) <-> (x2, y2), there is an equation:
        // A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        A(i, 0) = x1*x2;
        A(i, 1) = x1*y2;
        A(i, 2) = x1;
        A(i, 3) = y1*x2;
        A(i, 4) = y1*y2;
        A(i, 5) = y1;
        A(i, 6) = x2;
        A(i, 7) = y2;
        A(i, 8) = 1.0f;
    }

    // Fill last row of A with zeros for easier use of SVD
    for (int j = 0; j < 9; j++){
        A(8, j) = 0.0f;
    }

    return A;
}

// Pick k random matches
vector<Match> sampleRandomMatches(const vector<Match>& matches, int k) {
    vector<Match> random_matches;
    random_matches.reserve(k);
    vector<char> selected(matches.size(), 0);
    while (random_matches.size() < k) {
        int random_index = rand() % matches.size(); 
        // Avoid picking the same match twice
        if (!selected[random_index]) {
            random_matches.push_back(matches[random_index]);
            selected[random_index] = 1;
        }
    }
    return random_matches;
}

// 3x3 diagonal matrix for denormalization of F (diag(factor, factor, 1))
FMatrix<float,3,3> getNormalizationMatrix(){
    FMatrix<float, 3, 3> normalizationMatrix(0.0f);
    normalizationMatrix(0,0) = NORMALIZATION_FACTOR;
    normalizationMatrix(1,1) = NORMALIZATION_FACTOR;
    normalizationMatrix(2,2) = 1.0f;
    return normalizationMatrix;
}

// Compute current fundamental matrix from 8 point matches (x^TFx'=0)
FMatrix<float,3,3> getCurrentF(const vector<Match>& matches) {
    FMatrix<float, 3, 3> F;

    // Solve the equation Af=0 using SVD
    FMatrix<float,9,9> A = getA(matches);
    FVector<float,9> S;
    FMatrix<float,9,9> U, VT;
    svd(A, U, S, VT); // Compute SVD of A

    // F is the last row of VT (corresponding to smallest singular value)
    FVector<float,9> f = VT.getRow(8);
    F(0,0) = f[0]; F(0,1) = f[1]; F(0,2) = f[2];
    F(1,0) = f[3]; F(1,1) = f[4]; F(1,2) = f[5];
    F(2,0) = f[6]; F(2,1) = f[7]; F(2,2) = f[8];

    // Enforce rank 2 constraint on F by firstly computing SVD of F
    FVector<float,3> S2;
    FMatrix<float,3,3> U2, VT2;
    svd(F, U2, S2, VT2);

    // Put smallest singular value to 0 and recompose F with orthogonal projection
    S2[2] = 0.0f;
    F = U2 * Diagonal(S2) * VT2;

    // Denormalize F using F = N.T * F * N
    FMatrix<float,3,3> normalizationMatrix = getNormalizationMatrix();
    F = transpose(normalizationMatrix) * F * normalizationMatrix; //transpose doesn't have to be computed explicitly since N.T=N
    return F;
}

// Find inliers among matches for current F
vector<int> findInliers(const FMatrix<float,3,3>& F,
                       const vector<Match>& matches,
                       float distMax) {
    vector<int> inliers;
    for (int i = 0; i < matches.size(); i++) {
        FVector<float, 3> point = FVector<float, 3>(matches[i].x1, matches[i].y1, 1.0f);
        FVector<float, 3> pointPrime = FVector<float, 3>(matches[i].x2, matches[i].y2, 1.0F);

        // Compute distance from point to epipolar line (distance = |x'*l'|/sqrt(l'[0]^2+l'[1]^2))
        FVector<float, 3> epipolarLine = transpose(F) * point;
        float num = fabsf(pointPrime*epipolarLine);
        float denom = sqrt(epipolarLine[0]*epipolarLine[0] + epipolarLine[1]*epipolarLine[1]);
        
        if (denom < 1e-8f) // Avoid division by 0 and skip the current match
            continue;
        float distance = num/denom;

        // If the distance is smaller or eq than the treshold, the match is an inlier
        if (distance <= distMax){
            inliers.push_back(i);
        }
    }
    return inliers;
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter=100000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;
    // --------------- TODO ------------
    // DO NOT FORGET NORMALIZATION OF POINTS
    
    const int n = matches.size();
    const int k = 8; // Number of matches to compute F
    int currentNiter = 0;
    int m = 0; // Number of inliers of the best model

    while (currentNiter < Niter){
        vector<Match> random_matches = sampleRandomMatches(matches, k);
        FMatrix<float,3,3> F = getCurrentF(random_matches);
        vector<int> currentInliers = findInliers(F, matches, distMax);

        // If there are more inliers than before, update bestF, bestInliers and Niter
        if (currentInliers.size() > m) {
            bestF = F;
            bestInliers = currentInliers;
            m = bestInliers.size();

            double p = double(m)/double(n);
            // Escape too small p values
            if (p > 0.1) {
                Niter = (int)ceil((double)log(BETA)/(double)log(1 - pow(p, k)));
            }
        }

        ++currentNiter;
    }

    cout << "Number of total iterations " << currentNiter << "\n";
    cout << "Number of total inliers " << m << "\n";

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);

    // Check that inliers are correctly found
    // Uncomment to see the distance of the first inlier to its epipolar
    if (!bestInliers.empty()) {
        FVector<float,3> x(all[bestInliers[0]].x1, all[bestInliers[0]].y1, 1.0f);
        FVector<float,3> xp(all[bestInliers[0]].x2, all[bestInliers[0]].y2, 1.0f);
        float e = xp * (transpose(bestF) * x);
        cout << "Inlier distance from the epipolar line: " << e << "\n";
    }

    return bestF;
}

// Given line ax+by+c=0 and known x coordinate, return point (x,y) on the line
IntPoint2 getPointOnTheLine(int x, FVector<float, 3> line){
    // Avoid division by 0 and return trivial vertical line
    // if (std::fabs(line[1]) < 1e-8f) 
    //     return IntPoint2(x, 0);
    float y = -(line[0]*x + line[2])/line[1];
    return IntPoint2(x, (int)y);
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    while(true) {
        int x, y;
        if (getMouse(x, y) == 3)
            break;
         
        Color color(rand()%256,rand()%256,rand()%256); // Randomly sample a color for circle and line
        fillCircle(x, y, 2, color);

        FVector<float, 3> X, epipolarLine; 
        IntPoint2 point1, point2;
        int x1, x2;

        if (x < I1.width()) {
            // Clicked point is in I1, compute line in I2 (l'=F^T*X)
            X = FVector<float, 3>(x, y, 1);
            epipolarLine = transpose(F)*X; // returns a, b, c for line equation ax+by+c=0
            
            // Choose two x coordinates from the I2 and construct the points
            x1 = 0;
            x2 = I2.width()-1;
            point1 = getPointOnTheLine(x1, epipolarLine);
            point2 = getPointOnTheLine(x2, epipolarLine);
            
            // Shift line points by width of I1 to draw it in the second image
            point1.x() += I1.width();
            point2.x() += I1.width();
        } else {
            // Clicked point is in I2, compute line in I1 (l=F*X)
            X = FVector<float, 3>(x-I1.width(), y, 1); // Scale the x to I2 coordinates
            epipolarLine = F*X;
            
            // Choose two x coordinates from the I1 and construct the points
            x1 = 0;
            x2 = I1.width() - 1;
            point1 = getPointOnTheLine(x1, epipolarLine);
            point2 = getPointOnTheLine(x2, epipolarLine);
        }

        drawLine(point1, point2, color, 2);
    }
}

int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    // const char* s1 = argc > 1 ? argv[1]: srcPath("im1.jpg");
    // const char* s2 = argc > 2 ? argv[2]: srcPath("im2.jpg");

    std::string s1 = argc > 1 ? argv[1] : srcPath("im1.jpg");
    std::string s2 = argc > 2 ? argv[2] : srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    const int n = (int)matches.size();
    cout << " matches: " << n << endl;
    drawString(100,20,std::to_string(n)+ " matches",RED);
    click();

    FMatrix<float,3,3> F = computeF(matches);

    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    drawString(100, 20, to_string(matches.size())+"/"+to_string(n)+" inliers", RED);
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
