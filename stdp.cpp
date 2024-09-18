// To compile (adapt as needed):
// g++ -I $EIGEN_DIR/Eigen/ -O3 -std=c++11 stdp.cpp -o stdp

#include <string>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <ctime>
#include <Eigen/Dense>
#include <mmap_file.h>
#include <filesystem>
#include <assert.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define ASSERT(X) \
	do {                                                            \
		if (!(X)) {                                                 \
			assert(false);                                          \
			exit(EXIT_SUCCESS);                                     \
		}                                                           \
	} while(0)

#define NONE 0
#define LEARNING 1234
#define TESTING 4321
#define MIXING 5678
#define SPONTANEOUS 7913
#define PULSE 1978

//#define MOD (70.0 / 126.0)
#define MOD (1.0 / 126.0)

#define dt 1.0 // NOTE: Don't attempt to just modify the dt without reading the code below, as it will likely break things.

#define BASEALTD (14e-5 * 1.5 * 1.0)
#define RANDALTD 0.0
#define ALTP (8e-5  * .008  * 1.0 ) // Note ALTPMULT below
#define MINV -80.0
#define TAUVLONGTRACE 20000  
#define LATCONNMULTINIT 5.0  // The ALat coefficient; (12 * 36/100)

#define NBI 20
#define NBE 100

#define NBNEUR (NBE + NBI)

#define WFFINITMAX .1
#define WFFINITMIN 0.0
#define MAXW 50.0 
#define VSTIM 1.0

#define TIMEZEROINPUT 100
// #define NBPATTERNSLEARNING 500000
#define NBPATTERNSLEARNING 601
#define NBPATTERNSTESTING 1000 // 1000 
#define NBPATTERNSPULSE 50 
#define PRESTIMEMIXING 350 // in ms
#define PRESTIMEPULSE 350
#define NBPATTERNSSPONT 300
#define PRESTIMESPONT 1000
#define PULSESTART 0
//#define NBPRESPERPATTERNLEARNING 30 
#define NBMIXES 30

#define PATCHSIZE 17
#define FFRFSIZE (2 * PATCHSIZE * PATCHSIZE)

// Inhibition parameters
#define TAUINHIB 10 // in ms
#define ALPHAINHIB .6 // .6

#define NEGNOISERATE 0.0 // in KHz (expected number of thousands of VSTIM received per second through noise)
#define POSNOISERATE 1.8 // in KHz (expected number of thousands of VSTIM received per second through noise)

#define A 4
#define B .0805
#define Isp 400.0
#define TAUZ 40.0
#define TAUADAP 144.0
#define TAUVTHRESH 50.0
#define C 281.0
#define Gleak 30.0
#define Eleak -70.6
#define DELTAT 2.0  // in mV
#define VTMAX -30.4
#define VTREST -50.4
#define VPEAK 20  // Also in mV 
#define VRESET Eleak

// -70.5 is approximately the resting potential of the Izhikevich neurons, as it is of the AdEx neurons used in Clopath's experiments
#define VREST -70.5 


#define THETAVLONGTRACE  -45.3 // -45.3 //MINV // Eleak // VTMAX

#define MAXDELAYDT 20
#define NBSPIKINGSTEPS 1 // (3.0 / dt) // Number of steps that a spike should take - i.e. spiking time (in ms) / dt.
#define REFRACTIME  0 // (dt-.001)
#define THETAVPOS -45.3
#define THETAVNEG Eleak
#define TAUXPLAST 15.0 // all 'tau' constants are in ms
#define TAUVNEG 10.0 
#define TAUVPOS 7.0
#define VREF2 50 // 70  // in mV^2

#define NBNOISESTEPS 73333


using namespace Eigen;
using namespace std;

#define STRING(x) #x
#define XSTRING(x) STRING(x)

static const char *args_path = XSTRING(CMAKE_SOURCE_PATH) "/args.ini";

static
void
read_args_from_file(
	int *argc_ref,
	char *argv[],
	char *args_string
) {
	int argc = *argc_ref;
	uint8_t *mapped_data;
	size_t mapped_size;
	mmap_file_open_ro(args_path, &mapped_data, &mapped_size);
	char *mapped_ptr = (char*)mapped_data;
	char *end_ptr = 0;
	while (true) {
		char c = *mapped_ptr;
		if (c == ';') {
			mapped_ptr = strchr(mapped_ptr, '\n');
			mapped_ptr++;
			continue;
		}
		break;
	}
	end_ptr = strchr(mapped_ptr, '\n');
	if (end_ptr == NULL) {
		end_ptr = strchr(mapped_ptr, '\0');
	}
	strncpy(args_string, mapped_ptr, end_ptr - mapped_ptr);
	mmap_file_close(mapped_data, mapped_size);
	char *arg_ptr = args_string;
	while(true) {
		char c = *arg_ptr;
		if (c == '\0') break;
		if (c == ' ') {
			arg_ptr++;
			continue;
		}

		argv[argc++] = arg_ptr;
		while (true) {
			c = *++arg_ptr;
			if (c == ' ' || c == '\0') {
				*arg_ptr++ = '\0';
				break;
			}
		}
	}
	*argc_ref = argc;
}

MatrixXd poissonMatrix(const MatrixXd& lambd);
MatrixXd poissonMatrix2(const MatrixXd& lambd);
int poissonScalar(const double lambd);
void saveWeights(MatrixXd& wgt, string fn);
void readWeights(MatrixXd& wgt, string fn);

int main(int argc, char* argv[])
{
	srand(0);
	clock_t tic;
	int PHASE = NONE;
	int STIM1, STIM2;
	int PRESTIMELEARNING= 350; // ms
	int PRESTIMETESTING = 350;
	int PULSETIME = 100; // This is the time during which stimulus is active during PULSE trials (different from PRESTIMEPULSE which is total trial time)
	int NOLAT = 0;
	int NOELAT = 0;
	int NOINH = 0;
	int NOSPIKE = 0;
	int NONOISE = 0;
	int NBLASTSPIKESSTEPS =  0;
	int NBLASTSPIKESPRES = 50;
	int NBRESPS = -1;  // Number of resps (total nb of spike / total v for each presentation) to be stored in resps and respssumv. Must be set depending on the PHASE (learmning, testing, mixing, etc.)
	double LATCONNMULT = LATCONNMULTINIT;
	double INPUTMULT = -1.0; // To be modified!
	double DELAYPARAM = 5.0;

	char args_string[256] = {0};
	if (argc == 1) {
		read_args_from_file(&argc, argv, args_string);
	}

	if (argc == 1) {
		cerr << endl << "Error: You must provide at least one argument - 'learn', 'mix', 'pulse' or 'test'." << endl; 
		return -1;
	}

	if (strcmp(argv[1], "test") == 0)	PHASE = TESTING;
	if (strcmp(argv[1], "learn") == 0)	PHASE = LEARNING;
	if (strcmp(argv[1], "mix") == 0)	PHASE = MIXING;
	if (strcmp(argv[1], "pulse") == 0)	PHASE = PULSE;
	if (strcmp(argv[1], "spont") == 0)	PHASE = SPONTANEOUS;

	if (PHASE == NONE) { 
		cout << "Argument is invalid: '" << argv[1] << "'" << endl;
		return -1;
	}

	MatrixXd w = MatrixXd::Zero(NBNEUR, NBNEUR);
	MatrixXd wff = MatrixXd::Zero(NBNEUR, FFRFSIZE); 

	// These constants are only used for learning:
	double WPENSCALE = .33;
	double ALTPMULT = .75;
	double WEI_MAX = 20.0 * 4.32 / LATCONNMULT ; //1.5
	double WIE_MAX = .5 * 4.32 / LATCONNMULT;
	double WII_MAX = .5 * 4.32 / LATCONNMULT;
	int NBPATTERNS, PRESTIME, NBPRES, NBSTEPSPERPRES, NBSTEPS;

	// Command line parameters handling
	
	for (int nn=2; nn < argc; nn++){
		if (std::string(argv[nn]).compare("nonoise") == 0) {
			NONOISE = 1;
			cout << "No noise!" << endl;
		}
		if (std::string(argv[nn]).compare("nospike") == 0) {
			NOSPIKE = 1;
			cout << "No spiking! !" << endl;
		}
		if (std::string(argv[nn]).compare("noinh") == 0) {
			NOINH = 1;
			cout << "No inhibition!" << endl;
		}
		if (std::string(argv[nn]).compare("delayparam") == 0) {
			DELAYPARAM = std::stod(argv[nn+1]);
		}
		if (std::string(argv[nn]).compare("latconnmult") == 0) {
			LATCONNMULT = std::stod(argv[nn+1]);
		}
		if (std::string(argv[nn]).compare("wpenscale") == 0) {
			WPENSCALE = std::stod(argv[nn+1]);
		}
		if (std::string(argv[nn]).compare("timepres") == 0) {
			PRESTIMELEARNING = std::stoi(argv[nn+1]) ;
			PRESTIMETESTING = std::stoi(argv[nn+1]) ;
		}
		if (std::string(argv[nn]).compare("altpmult") == 0) {
			ALTPMULT = std::stod(argv[nn+1]) ;
		}
		if (std::string(argv[nn]).compare("wie") == 0) {
			WIE_MAX = std::stod(argv[nn+1]) * 4.32 / LATCONNMULT;
			// WII max is yoked to WIE max
			WII_MAX = std::stod(argv[nn+1]) * 4.32 / LATCONNMULT;
		}
		if (std::string(argv[nn]).compare("wei") == 0) {
			WEI_MAX = std::stod(argv[nn+1]) * 4.32 / LATCONNMULT;
		}
		if (std::string(argv[nn]).compare("nolat") == 0) {
			NOLAT = 1;
			cout << "No lateral connections! (Either E or I)" << endl;
		}
		if (std::string(argv[nn]).compare("noelat") == 0) {
			NOELAT = 1;
			cout << "No E-E lateral connections! (E-I, I-I and I-E unaffected)" << endl;
		}
		if (std::string(argv[nn]).compare("pulsetime") == 0) {
			PULSETIME = std::stoi(argv[nn+1]);
		}
	}

// On the command line, you must specify one of 'learn', 'pulse', 'test', 'spontaneous', or 'mix'. If using 'pulse', you must specify a stimulus number. IF using 'mix', you must specify two stimulus numbers.

	if (PHASE == LEARNING) {
		NBPATTERNS = NBPATTERNSLEARNING; PRESTIME = PRESTIMELEARNING; NBPRES = NBPATTERNS; //* NBPRESPERPATTERNLEARNING;
		NBLASTSPIKESPRES = 30;
		NBRESPS = 2000;
		w =  MatrixXd::Zero(NBNEUR, NBNEUR); //MatrixXd::Random(NBNEUR, NBNEUR).cwiseAbs();
		//w.fill(1.0);
		w.bottomRows(NBI).leftCols(NBE).setRandom(); // Inhbitory neurons receive excitatory inputs from excitatory neurons
		w.rightCols(NBI).setRandom(); // Everybody receives inhibition (including inhibitory neurons)
		w.bottomRows(NBI).rightCols(NBI) =  -w.bottomRows(NBI).rightCols(NBI).cwiseAbs() * WII_MAX;
		w.topRows(NBE).rightCols(NBI) = -w.topRows(NBE).rightCols(NBI).cwiseAbs() * WIE_MAX;
		w.bottomRows(NBI).leftCols(NBE) = w.bottomRows(NBI).leftCols(NBE).cwiseAbs() * WEI_MAX;
		w = w - w.cwiseProduct(MatrixXd::Identity(NBNEUR, NBNEUR)); // Diagonal lateral weights are 0 (no autapses !)
		wff =  (WFFINITMIN + (WFFINITMAX-WFFINITMIN) * MatrixXd::Random(NBNEUR, FFRFSIZE).cwiseAbs().array()).cwiseMin(MAXW) ; //MatrixXd::Random(NBNEUR, NBNEUR).cwiseAbs();
		wff.bottomRows(NBI).setZero(); // Inhibitory neurons do not receive FF excitation from the sensory RFs (should they? TRY LATER)
	} else if (PHASE == PULSE) {
		NBPATTERNS = NBPATTERNSPULSE; PRESTIME = PRESTIMEPULSE; NBPRES = NBPATTERNS; //* NBPRESPERPATTERNTESTING;
		if (argc < 3) { cerr << endl << "Error: When using 'pulse', you must provide the number of the stimulus you want to pulse." << endl; return -1; }
		STIM1 = std::stoi(argv[2]) -1; // -1 because of c++ zero-counting (the nth pattern has location n-1 in the array)
		NBLASTSPIKESPRES = NBPATTERNS;
		NBRESPS = NBPRES;
		cout << "Stim1: " << STIM1 << endl;
		readWeights(w, "w.dat");
		readWeights(wff, "wff.dat");
		cout << "Pulse input time: " << PULSETIME << " ms" << endl;
	} else if (PHASE == TESTING) {
		NBPATTERNS = NBPATTERNSTESTING; PRESTIME = PRESTIMETESTING; NBPRES = NBPATTERNS; //* NBPRESPERPATTERNTESTING;
		NBLASTSPIKESPRES = 30;
		NBRESPS = NBPRES;
		readWeights(w, "w.dat");
		readWeights(wff, "wff.dat");
		cout << "First row of w (lateral weights): " << w.row(0) << endl;
		cout << "w(1,2) and w(2,1): " << w(1,2) << " " << w(2,1) << endl;
		
		//w.bottomRows(NBI).leftCols(NBE).fill(1.0); // Inhbitory neurons receive excitatory inputs from excitatory neurons
		//w.rightCols(NBI).fill(-1.0); // Everybody receives fixed, negative inhibition (including inhibitory neurons)
	} else if (PHASE == SPONTANEOUS) {
		NBPATTERNS = NBPATTERNSSPONT; PRESTIME = PRESTIMESPONT; NBPRES = NBPATTERNS; //* NBPRESPERPATTERNTESTING;
		NBLASTSPIKESPRES = NBPATTERNS;
		NBRESPS = NBPRES;
		readWeights(w, "w.dat");
		readWeights(wff, "wff.dat");
		cout << "Spontaneous activity - no stimulus !" << endl;
	} else if (PHASE == MIXING) {
		NBPATTERNS = 2; PRESTIME = PRESTIMEMIXING ; NBPRES = NBMIXES * 3; //* NBPRESPERPATTERNTESTING;
		NBLASTSPIKESPRES = 30;
		NBRESPS = NBPRES;
		readWeights(w, "w.dat");
		readWeights(wff, "wff.dat");
		if (argc < 4) { cerr << endl << "Error: When using 'mix', you must provide the numbers of the 2 stimuli you want to mix." << endl; return -1; }
		STIM1 = std::stoi(argv[2]) -1;
		STIM2 = std::stoi(argv[3]) -1;
		cout << "Stim1, Stim2: " << STIM1 << ", " << STIM2 <<endl;
	} else { 
		cerr << "Which phase?\n"; return -1;
	}
	cout << "Lat. conn.: " << LATCONNMULT << endl;
	cout << "WIE_MAX: " << WIE_MAX << " / " << WIE_MAX *LATCONNMULT / 4.32 <<  endl;
	cout << "DELAYPARAM: " << DELAYPARAM  << endl;
	cout << "WPENSCALE: " << WPENSCALE << endl;
	cout << "ALTPMULT: " << ALTPMULT<< endl;
	NBSTEPSPERPRES = (int)(PRESTIME / dt);
	NBLASTSPIKESSTEPS = NBLASTSPIKESPRES * NBSTEPSPERPRES; 
	NBSTEPS = NBSTEPSPERPRES * NBPRES;

	MatrixXi lastnspikes = MatrixXi::Zero(NBNEUR, NBLASTSPIKESSTEPS);
	MatrixXd lastnv = MatrixXd::Zero(NBNEUR, NBLASTSPIKESSTEPS);

	cout << "Reading input data...." << endl;
	int nbpatchesinfile = 0, totaldatasize = 0;

	// The stimulus patches are 17x17x2 in length, arranged linearly. See below for the setting of feedforward firing rates based on patch data.
	// See also  makepatchesImageNetInt8.m

	ifstream DataFile ("../patchesCenteredScaledBySumTo126ImageNetONOFFRotatedNewInt8.bin.dat", ios::in | ios::binary | ios::ate);
	if (!DataFile.is_open()) { 
		throw ios_base::failure("Failed to open the binary data file!");
		return -1;
	}
	ifstream::pos_type  fsize = DataFile.tellg();
	char *membuf = new char[fsize];
	DataFile.seekg (0, ios::beg);
	DataFile.read(membuf, fsize);
	DataFile.close();
	int8_t* imagedata = (int8_t*) membuf;
	//double* imagedata = (double*) membuf;
	cout << "Data read!" << endl;
	//totaldatasize = fsize / sizeof(double); // To change depending on whether the data is float/single (4) or double (8)
	totaldatasize = fsize / sizeof(int8_t); // To change depending on whether the data is float/single (4) or double (8)
	nbpatchesinfile = totaldatasize / (PATCHSIZE * PATCHSIZE) - 1; // The -1 is just there to ignore the last patch (I think)
	cout << "Total data size (number of values): " << totaldatasize << ", number of patches in file: " << nbpatchesinfile << endl;
	cout << imagedata[5654] << " " << imagedata[6546] << " " << imagedata[9000] << endl;
	
	// The noise excitatory input is a Poisson process (separate for each cell) with a constant rate (in KHz / per ms)
	// We store it as "frozen noise" to save time.
	MatrixXd negnoisein = - poissonMatrix(dt * MatrixXd::Constant(NBNEUR, NBNOISESTEPS, NEGNOISERATE)) * VSTIM;
	MatrixXd posnoisein = poissonMatrix(dt * MatrixXd::Constant(NBNEUR, NBNOISESTEPS, POSNOISERATE)) * VSTIM;
	if (NONOISE || NOSPIKE) { // If No-noise or no-spike, suppress the background bombardment of random I and E spikes
		posnoisein.setZero();
		negnoisein.setZero();
	}

	VectorXd v_arr =  VectorXd::Constant(NBNEUR, VREST); // VectorXd::Zero(NBNEUR); // -70.5 is approximately the resting potential of the Izhikevich neurons, as it is of the AdEx neurons used in Clopath's experiments

	// Initializations. 
	VectorXd xplast_ff_arr = VectorXd::Zero(FFRFSIZE);
	VectorXd xplast_lat_arr = VectorXd::Zero(NBNEUR);
	VectorXd vneg_arr = v_arr;
	VectorXd vpos_arr = v_arr;

	// Correct initialization for vlongtrace.
	VectorXd vlongtrace_arr = VectorXd::Constant(NBNEUR, VREST - THETAVLONGTRACE).cwiseMax(0);

	VectorXd z_arr = VectorXd::Zero(NBNEUR);
	VectorXd wadap_arr = VectorXd::Zero(NBNEUR);
	VectorXd vthresh_arr = VectorXd::Constant(NBNEUR, VTREST);
	VectorXi isspiking_arr = VectorXi::Zero(NBNEUR);

	double ALTDS[NBNEUR]; 
	for (int nn = 0; nn < NBNEUR; nn++) {
		ALTDS[nn] = BASEALTD + RANDALTD*( (double)rand() / (double)RAND_MAX );
	}

	VectorXd lgnrates = VectorXd::Zero(FFRFSIZE);
	VectorXd lgnratesS1 = VectorXd::Zero(FFRFSIZE);
	VectorXd lgnratesS2 = VectorXd::Zero(FFRFSIZE);
	VectorXd lgnfirings_arr = VectorXd::Zero(FFRFSIZE);

	VectorXd sumwff = VectorXd::Zero(NBPRES);
	VectorXd sumw = VectorXd::Zero(NBPRES);
	MatrixXi resps = MatrixXi::Zero(NBNEUR, NBRESPS);
	MatrixXd respssumv = MatrixXd::Zero(NBNEUR, NBRESPS);
	
	double mixvals[NBMIXES];
	for (int nn = 0; nn < NBMIXES; nn++) {
		// NBMIXES values equally spaced from 0 to 1 inclusive.
		mixvals[nn] = (double)nn / (double)(NBMIXES - 1); 
	}

	fstream myfile;
			
	// If no-inhib mode, remove all inhibitory connections:
	if (NOINH) {
		w.rightCols(NBI).setZero();
	}

	// We generate the delays:

	// We use a trick to generate an exponential distribution, median should be small (maybe 2-4ms)
	// The mental image is that you pick a uniform value in the unit line,
	//repeatedly check if it falls below a certain threshold - if not, you cut
	//out the portion of the unit line below that threshold and stretch the
	//remainder (including the random value) to fill the unit line again. Each time you increase a counter, stopping when the value finally falls below the threshold. The counter at the end of this process has exponential distribution.
	// There's very likely simpler ways to do it.

	// DELAYPARAM should be a small value (3 to 6). It controls the median of the exponential.
	int delays[NBNEUR][NBNEUR];
	for (int ni = 0; ni < NBNEUR; ni++) {
		for (int nj = 0; nj < NBNEUR; nj++) {
			double val = (double)rand() / (double)RAND_MAX;
			double crit= 1.0 / DELAYPARAM; // .1666666666;
			int mydelay = 1;
			for (; mydelay <= MAXDELAYDT; mydelay++) {
				if (val < crit) break;
				val = DELAYPARAM  * (val - crit) / (DELAYPARAM -1.0) ; // "Cutting" and "Stretching"
			}
			if (mydelay > MAXDELAYDT) mydelay = 1;
			delays[nj][ni] = mydelay;
		}
	}
 
	// NOTE: We implement the machinery for feedforward delays, but they are NOT used (see below).   
	//myfile.open("delays.txt", ios::trunc | ios::out); 
	// int delaysFF[FFRFSIZE][NBNEUR];
	// VectorXi incomingFFspikes[NBNEUR][FFRFSIZE];  
	// for (int ni = 0; ni < NBNEUR; ni++) {
	// 	for (int nj = 0; nj < FFRFSIZE; nj++) {
	// 		double val = (double)rand() / (double)RAND_MAX;
	// 		double crit = 0.2;
	// 		int mydelay = 1;
	// 		for (; mydelay <= MAXDELAYDT; mydelay++) {
	// 			if (val < crit) break;
	// 			val = 5.0 * (val - crit) / 4.0 ;
	// 		}
	// 		if (mydelay > MAXDELAYDT) mydelay = 1;
	// 		delaysFF[nj][ni] = mydelay;
	// 		incomingFFspikes[ni][nj] = VectorXi::Zero(mydelay); 
	// 	}
	// }

	// Keeping this to keep seed the same (still matching original)
	for (int ni = 0; ni < NBNEUR; ni++) {
		for (int nj = 0; nj < FFRFSIZE; nj++) {
			rand();
		}
	}

	// Initializations done, let's get to it!

	tic = clock();
	int numstep = 0;

	// For each stimulus presentation...
	for (int numpres = 0; numpres < NBPRES; numpres++) {
		// Where are we in the data file?
		int posindata = ((numpres % nbpatchesinfile) * FFRFSIZE / 2 );
		if (PHASE == PULSE) {
			posindata = ((STIM1 % nbpatchesinfile) * FFRFSIZE / 2 );
		}

		if (posindata >= totaldatasize - FFRFSIZE / 2) { 
			cerr << "Error: tried to read beyond data end.\n";
			return -1; 
		}

		// Extracting the image data for this frame presentation, and preparing the LGN / FF output rates (notice the log-transform):
		
		for (int nn=0; nn < FFRFSIZE / 2; nn++) {
			lgnrates(nn) = log(1.0 + ((double)imagedata[posindata+nn] > 0 ? MOD * (double)imagedata[posindata+nn] : 0));
			lgnrates(nn + FFRFSIZE / 2) = log(1.0 + ((double)imagedata[posindata+nn] < 0 ? - MOD * (double)imagedata[posindata+nn] : 0));
		}
		lgnrates /= lgnrates.maxCoeff(); // Scale by max! The inputs are scaled to have a maximum of 1 (multiplied by INPUTMULT below)

		if (PHASE == MIXING) {
			int posindata1 = ((STIM1 % nbpatchesinfile) * FFRFSIZE / 2 ); if (posindata1 >= totaldatasize - FFRFSIZE / 2) { cerr << "Error: tried to read beyond data end.\n"; return -1; }
			int posindata2 = ((STIM2 % nbpatchesinfile) * FFRFSIZE / 2 ); if (posindata2 >= totaldatasize - FFRFSIZE / 2) { cerr << "Error: tried to read beyond data end.\n"; return -1; }

			double mixval1 = mixvals[numpres % NBMIXES]; double mixval2 = 1.0-mixval1; double mixedinput = 0;
			if ((numpres / NBMIXES) == 1 ) mixval2 = 0;
			if ((numpres / NBMIXES) == 2 ) mixval1 = 0;
			
			for (int nn = 0; nn < FFRFSIZE / 2; nn++) {
				lgnratesS1(nn) = log(1.0 + ((double)imagedata[posindata1+nn] > 0 ? (double)imagedata[posindata1+nn] : 0));  lgnratesS1(nn + FFRFSIZE / 2) = log(1.0 + ((double)imagedata[posindata1+nn] < 0 ? -(double)imagedata[posindata1+nn] : 0));
				lgnratesS2(nn) = log(1.0 + ((double)imagedata[posindata2+nn] > 0 ? (double)imagedata[posindata2+nn] : 0));  lgnratesS2(nn + FFRFSIZE / 2) = log(1.0 + ((double)imagedata[posindata2+nn] < 0 ? -(double)imagedata[posindata2+nn] : 0));
				// No log-transform:
				//lgnratesS1(nn) = ( (imagedata[posindata1+nn] > 0 ? imagedata[posindata1+nn] : 0));  lgnratesS1(nn + FFRFSIZE / 2) = ( (imagedata[posindata1+nn] < 0 ? -imagedata[posindata1+nn] : 0));
				//lgnratesS2(nn) = ( (imagedata[posindata2+nn] > 0 ? imagedata[posindata2+nn] : 0));  lgnratesS2(nn + FFRFSIZE / 2) = ( (imagedata[posindata2+nn] < 0 ? -imagedata[posindata2+nn] : 0));
			}
			lgnratesS1 /= lgnratesS1.maxCoeff(); // Scale by max!!
			lgnratesS2 /= lgnratesS2.maxCoeff(); // Scale by max!!
			
			for (int nn = 0; nn < FFRFSIZE; nn++) {
				lgnrates(nn) = mixval1 * lgnratesS1(nn) + mixval2 * lgnratesS2(nn);
			}
			
		}

		INPUTMULT = 150.0;
		INPUTMULT *= 2.0;
		
		lgnrates *= INPUTMULT; // We put inputmult here to ensure that it is reflected in the actual number of incoming spikes

		lgnrates *= (dt/1000.0);  // LGN rates from the pattern file are expressed in Hz. We want it in rate per dt, and dt itself is expressed in ms.

		// At the beginning of every presentation, we reset everything ! (it is important for the random-patches case which tends to generate epileptic self-sustaining firing; 'normal' learning doesn't need it.)
		v_arr.fill(Eleak);
		resps.col(numpres % NBRESPS).setZero();
		lgnfirings_arr.setZero();
		int incoming_spikes[NBNEUR][NBNEUR] = {0};

		// Stimulus presentation
		for (int numstepthispres = 0; numstepthispres < NBSTEPSPERPRES; numstepthispres++) {
			// We determine FF spikes, based on the specified lgnrates:

			// In the PULSE case, inputs only fire for a short period of time
			const bool is_pulse_cond = PHASE == PULSE && numstepthispres >= (double)(PULSESTART)/dt && numstepthispres < (double)(PULSESTART + PULSETIME)/dt;

			// Otherwise, inputs only fire until the 'relaxation' period at the end of each presentation
			const bool is_not_pulse_cond = PHASE != PULSE && numstepthispres < NBSTEPSPERPRES - ((double)TIMEZEROINPUT / dt);

			if (PHASE != SPONTANEOUS && (is_pulse_cond || is_not_pulse_cond)) { 
				for (int nn=0; nn < FFRFSIZE; nn++) {
					lgnfirings_arr(nn) = (rand() / (double)RAND_MAX < abs(lgnrates(nn)) ? 1.0 : 0.0); // Note that this may go non-poisson if the specified lgnrates are too high (i.e. not << 1.0)
				}
			} else {
				lgnfirings_arr.setZero();
			}

			xplast_ff_arr = xplast_ff_arr + lgnfirings_arr / TAUXPLAST - (dt / TAUXPLAST) *  xplast_ff_arr;

			// We compute the feedforward input: (see original comments)
			VectorXd LatInput = VectorXd::Zero(NBNEUR);
			for (int ni = 0; ni < NBNEUR; ni++) {
				for (int nj = 0; nj< NBNEUR; nj++) {
					if (NOELAT && nj < 100 && ni < 100) continue; // If NOELAT, E-E synapses are disabled.
					if (ni == nj) continue; // No autapses

					LatInput(ni) += w(ni, nj) * (incoming_spikes[ni][nj] & 1);
				}
			}

			VectorXd Iff = wff * lgnfirings_arr * VSTIM;
			VectorXd Ilat = LATCONNMULT * VSTIM * LatInput;

			// This disables all lateral connections - Inhibitory and excitatory
			if (NOLAT) {
				Ilat.setZero();
			}

			// Total input (FF + lateral + frozen noise):
			VectorXd I_arr = Iff + Ilat + posnoisein.col(numstep % NBNOISESTEPS) + negnoisein.col(numstep % NBNOISESTEPS);  //- InhibVect;

			VectorXi firings_arr = VectorXi::Zero(NBNEUR);
			for (int nn = 0; nn < NBNEUR; nn++) {
				double I = I_arr(nn);
				double vlongtrace = vlongtrace_arr(nn);
				double v = v_arr(nn);
				double vneg = vneg_arr(nn);
				double vpos = vpos_arr(nn);
				double z = z_arr(nn);
				double wadap = wadap_arr(nn);
				double vthresh = vthresh_arr(nn);
				int is_spiking = isspiking_arr(nn);
				double xplast_lat = xplast_lat_arr(nn);

				vlongtrace += (dt / TAUVLONGTRACE) * (MAX(0, v - THETAVLONGTRACE) - vlongtrace);
				vlongtrace = MAX(0, vlongtrace);

				vneg += (dt / TAUVNEG) * (v - vneg);
				vpos += (dt / TAUVPOS) * (v - vpos);

				if (NOSPIKE) {
					v += (dt/C) * (-Gleak * (v-Eleak) + z - wadap ) + I;
				} else {
					v += (dt/C) * (-Gleak * (v-Eleak) + z - wadap + Gleak * DELTAT * exp((v - vthresh) / DELTAT ) ) + I;
				}

				v = is_spiking > 0 ? VPEAK - 0.001 : v;
				v = is_spiking == 1 ? VRESET : v;
				v = MAX(MINV, v);

				z = is_spiking == 1 ? Isp : z;
				vthresh = is_spiking == 1 ? VTMAX : vthresh;
				wadap = is_spiking == 1 ? wadap + B : wadap;

				is_spiking = MAX(0, is_spiking - 1);

				int firings = 0;
				if (!NOSPIKE) {
					firings = v > VPEAK ? 1 : 0;
					v = firings > 0 ? VPEAK : v;
					is_spiking = firings > 0 ? NBSPIKINGSTEPS : is_spiking;
					if (firings) {
						for (int nj=0; nj < NBNEUR; nj++) {
							int fire = nj != nn;

							incoming_spikes[nj][nn] |= fire << delays[nn][nj];
						}
					}
				}

				wadap =  wadap + (dt / TAUADAP) * (A * (v - Eleak) - wadap);
				z = z + (dt / TAUZ) * -1.0 * z;
				vthresh = vthresh + (dt / TAUVTHRESH) * (-1.0 * vthresh + VTREST);

				xplast_lat = xplast_lat + (double)firings / TAUXPLAST - (dt / TAUXPLAST) * xplast_lat;

				vlongtrace_arr(nn) = vlongtrace;
				v_arr(nn) = v;
				vneg_arr(nn) = vneg;
				vpos_arr(nn) = vpos;
				z_arr(nn) = z;
				vthresh_arr(nn) = vthresh;
				wadap_arr(nn) = wadap;
				isspiking_arr(nn) = is_spiking;
				firings_arr(nn) = firings;
				xplast_lat_arr(nn) = xplast_lat;
			}

			// Plasticity !
			if (PHASE == LEARNING && numpres >= 401) {
				// For each neuron, we compute the quantities by which any synapse reaching this given neuron should be modified, if the synapse's firing / recent activity (xplast) commands modification.
				for (int nn = 0; nn < NBE; nn++) {
					double eachNeurLTD =  dt * (-ALTDS[nn] / VREF2) * vlongtrace_arr(nn) * vlongtrace_arr(nn) * MAX(0, vneg_arr(nn) - THETAVNEG);
					double eachNeurLTP =  dt * ALTP  * ALTPMULT * MAX(0, vpos_arr(nn) - THETAVNEG) * MAX(0, v_arr(nn) - THETAVPOS);

					for (int syn = 0; syn < FFRFSIZE; syn++) {
						double lgnfirings_mul = lgnfirings_arr(syn) > 1e-10 ? 1.0 : 0.0;
						wff(nn, syn) += xplast_ff_arr(syn) * eachNeurLTP;
						wff(nn, syn) += lgnfirings_mul * eachNeurLTD * (1.0 + wff(nn,syn) * WPENSCALE);
					}

					for (int syn = 0; syn < NBE; syn++) {
						double incoming_spikes_mul = incoming_spikes[nn][syn] & 1 > 0 ? 1.0 : 0.0;
						w(nn, syn) += xplast_lat_arr(syn) * eachNeurLTP;
						w(nn, syn) += incoming_spikes_mul * eachNeurLTD * (1.0 + w(nn,syn) * WPENSCALE);
					}
				}
				w.diagonal().setZero();
				wff = wff.cwiseMax(0);
				w.leftCols(NBE) = w.leftCols(NBE).cwiseMax(0);
				w.rightCols(NBI) = w.rightCols(NBI).cwiseMin(0);
				wff = wff.cwiseMin(MAXW);
				w = w.cwiseMin(MAXW);
			}

			for (int ni = 0; ni < NBNEUR; ni++) {
				for (int nj = 0; nj< NBNEUR; nj++) {
					incoming_spikes[nj][ni] = incoming_spikes[nj][ni] >> 1;
				}
			}

			// Storing some indicator variablkes...

			resps.col(numpres % NBRESPS) += firings_arr;
			respssumv.col(numpres % NBRESPS) += v_arr.cwiseMin(VTMAX); // We only record subthreshold potentials ! 
			lastnspikes.col(numstep % NBLASTSPIKESSTEPS) = firings_arr;
			lastnv.col(numstep % NBLASTSPIKESSTEPS) = v_arr;

			// Tempus fugit.
			numstep++;
		}

		sumwff(numpres) = wff.sum();
		sumw(numpres) = w.sum();
		if (numpres % 100 == 0) {
			cout << "Presentation " << numpres << " / " << NBPRES << endl; 
			cout << "TIME: " << (double)(clock() - tic) / (double)CLOCKS_PER_SEC <<endl;
			tic = clock();
			cout << "Total spikes for each neuron for this presentation: " << resps.col(numpres % NBRESPS).transpose() << endl;
			cout << "Vlongtraces: " << vlongtrace_arr.transpose() << endl;
			cout << " Max LGN rate (should be << 1.0): " << lgnrates.maxCoeff() << endl;
		}
		if ((numpres + 1) % 10000 == 0 || numpres + 1 == NBPRES) {
			std::string nolatindicator ("");
			std::string noinhindicator ("");
			std::string nospikeindicator ("");
			if (NOINH) noinhindicator = "_noinh";
			if (NOSPIKE) nospikeindicator = "_nospike";
			if (NOLAT) nolatindicator = "_nolat";
			if (NOELAT) nolatindicator = "_noelat";

			myfile.open("lastnspikes"+nolatindicator+".txt", ios::trunc | ios::out);  myfile << endl << lastnspikes << endl; myfile.close();

			switch(PHASE) {
				case TESTING: {
					myfile.open("resps_test.txt", ios::trunc | ios::out);  myfile << endl << resps << endl; myfile.close();
					myfile.open("lastnv_test"+nolatindicator+noinhindicator+".txt", ios::trunc | ios::out);  myfile << endl << lastnv << endl; myfile.close();
				} break;
				case SPONTANEOUS: {
					myfile.open("resps_spont.txt", ios::trunc | ios::out);  myfile << endl << resps << endl; myfile.close();
					myfile.open("lastnspikes_spont"+nolatindicator+noinhindicator+".txt", ios::trunc | ios::out);  myfile << endl << lastnspikes << endl; myfile.close();
				} break;
				case PULSE: {
					myfile.open("resps_pulse"+nolatindicator+noinhindicator+".txt", ios::trunc | ios::out);  myfile << endl << resps << endl; myfile.close();
					myfile.open("resps_pulse_"+std::to_string((long long int)STIM1)+".txt", ios::trunc | ios::out);  myfile << endl << resps << endl; myfile.close();
					myfile.open("lastnspikes_pulse"+nolatindicator+noinhindicator+".txt", ios::trunc | ios::out);  myfile << endl << lastnspikes << endl; myfile.close();
					myfile.open("lastnspikes_pulse_"+std::to_string((long long int)STIM1)+nolatindicator+noinhindicator+".txt", ios::trunc | ios::out);  myfile << endl << lastnspikes << endl; myfile.close();
				} break;
				case MIXING: {
					myfile.open("respssumv_mix" + nolatindicator + noinhindicator + nospikeindicator + ".txt", ios::trunc | ios::out);  myfile << endl << respssumv << endl; myfile.close();
					myfile.open("resps_mix" + nolatindicator +noinhindicator + nospikeindicator +  ".txt", ios::trunc | ios::out);  myfile << endl << resps << endl; myfile.close();
					myfile.open("respssumv_mix"+std::to_string((long long int)STIM1)+"_"+std::to_string((long long int)STIM2) + nolatindicator + noinhindicator + nospikeindicator + ".txt", ios::trunc | ios::out);  myfile << endl << respssumv << endl; myfile.close();
					myfile.open("resps_mix_"+std::to_string((long long int)STIM1)+"_"+std::to_string((long long int)STIM2) + nolatindicator + noinhindicator + nospikeindicator + ".txt", ios::trunc | ios::out);  myfile << endl << resps << endl; myfile.close();
				} break;
				case LEARNING: {
					cout << "(Saving temporary data ... )" << endl;

					myfile.open("w.txt", ios::trunc | ios::out);
					myfile << endl << w << endl;
					myfile.close();
					myfile.open("wff.txt", ios::trunc | ios::out);  myfile << endl << wff << endl; myfile.close();
					if ((numpres +1)%  50000 == 0) 
					{ 
						char tmpstr[80]; 
						sprintf(tmpstr, "%s%d%s", "wff_", numpres+1, ".txt"); myfile.open(tmpstr, ios::trunc | ios::out);  myfile << endl << wff << endl; myfile.close(); 
						sprintf(tmpstr, "%s%d%s", "w_", numpres+1, ".txt"); myfile.open(tmpstr, ios::trunc | ios::out);  myfile << endl << w << endl; myfile.close(); 
						saveWeights(w, "w_" + std::to_string( (long long int) (numpres+1)) +".dat");
						saveWeights(wff, "wff_" + std::to_string( (long long int) (numpres+1)) +".dat");
					}
					myfile.open("resps.txt", ios::trunc | ios::out);
					myfile << endl << resps << endl;
					myfile.close();

					saveWeights(w, "w.dat");
					saveWeights(wff, "wff.dat");
				} break;
			}
		}
	}
}

/*
 *  Utility functions
 */

MatrixXd poissonMatrix2(const MatrixXd& lambd) {
	MatrixXd k = MatrixXd::Zero(lambd.rows(),lambd.cols());
	for (int nr = 0; nr < lambd.rows(); nr++)
		for (int nc = 0; nc < lambd.cols(); nc++)
			k(nr, nc) = poissonScalar(lambd(nr, nc));
	return k;
}

MatrixXd poissonMatrix(const MatrixXd& lambd) {
	//MatrixXd lambd = MatrixXd::Random(SIZ,SIZ).cwiseAbs();
	//MatrixXd lambd = MatrixXd::Random(SIZ,SIZ).cwiseMax(0);
	//MatrixXd lambd = MatrixXd::Constant(SIZ,SIZ, .5);

	MatrixXd L = (-1 * lambd).array().exp();
	MatrixXd k = MatrixXd::Zero(lambd.rows(),lambd.cols());
	MatrixXd p = MatrixXd::Constant(lambd.rows(),lambd.cols(),1.0);
	MatrixXd matselect = MatrixXd::Constant(lambd.rows(),lambd.cols(),1.0);

	while ((matselect.array() > 0).any()) {
		k = (matselect.array() > 0).select(k.array() + 1, k); // wherever p > L (after the first loop, otherwise everywhere), k += 1
		p = p.cwiseProduct(MatrixXd::Random(p.rows(),p.cols()).cwiseAbs());  // p = p * random[0,1]
		matselect = (p.array() > L.array()).select(matselect, -1.0);
	}

	k = k.array() - 1;
	return k;
}
/* 
 // Test code for poissonMatrix:
	double SIZ=19;
	srand(time(NULL));
	MatrixXd lbd;
	MatrixXd kout;
	double dd = 0;
	for (int nn = 0; nn < 10000; nn++)
	{
		lbd = MatrixXd::Random(SIZ,SIZ).cwiseMax(0);
		kout = poissonMatrix(lbd);
	   dd += lbd.sum();
	}
	cout << endl << lbd << endl;
	cout << endl << kout << endl;
	cout << kout.mean() << endl;
	cout << dd << endl;
*/


int poissonScalar(const double lambd) {
	double L = exp(-1 * lambd);
	int k =  0; double p = 1.0;
	do{
		k = k + 1;
		p = p * (double)rand() / (double)RAND_MAX;
	} 
	while(p > L);
	return (k-1);
}

void saveWeights(MatrixXd& wgt, string fname) {
	double wdata[wgt.rows() * wgt.cols()];
	int idx=0;
	//cout << endl << "Saving weights..." << endl;
	for (int cc=0; cc < wgt.cols(); cc++)
		for (int rr=0; rr < wgt.rows(); rr++)
			wdata[idx++] = wgt(rr, cc);

	ofstream myfile(fname, ios::binary | ios::trunc);
	if (!myfile.write((char*) wdata, wgt.rows() * wgt.cols() * sizeof(double)))
		throw std::runtime_error("Error while saving matrix of weights.\n");
	myfile.close();
}

void readWeights(MatrixXd& wgt, string fname) {
	double wdata[wgt.cols() * wgt.rows()];
	int idx=0;
	cout << endl << "Reading weights from file " << fname << endl;
	ifstream myfile(fname, ios::binary);
	if (!myfile.read((char*) wdata, wgt.cols() * wgt.rows() * sizeof(double)))
		throw std::runtime_error("Error while reading matrix of weights.\n");
	myfile.close();
	for (int cc=0; cc < wgt.cols() ; cc++)
		for (int rr=0; rr < wgt.rows(); rr++)
			wgt(rr, cc) = wdata[idx++];
	cout << "Done!" <<endl;
}

