/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare,
 *  Matthew Hausknecht, and the Reinforcement Learning and Artificial Intelligence
 *  Laboratory
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  sharedLibraryInterfaceExample.cpp
 *
 *  Sample code for running an agent with the shared library interface.
 **************************************************************************** */

#include <iostream>
#include <ale_interface.hpp>

#ifdef __USE_SDL
  #include <SDL.h>
#endif
#include <time.h>
#include <sys/time.h>
#include <omp.h>

using namespace std;


double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main(int argc, char** argv) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " rom_file nthreads" << std::endl;
        return 1;
    }
    // cout << argv[1] << endl;
    // cout << argv[2] << endl;

    int num_procs = omp_get_num_procs();
    cout << "OMP Num Procs: " << num_procs << endl;
    int nthread =  atoi(argv[2]);
    cout << "Num threads: " << nthread << endl;
    ALEInterface ale[nthread];

    for (int i = 0; i < nthread; i++ ){
        // Get & Set the desired settings
        ale[i].setInt("random_seed", 123);
        //The default is already 0.25, this is just an example
        ale[i].setFloat("repeat_action_probability", 0.25);

        #ifdef __USE_SDL
            ale[i].setBool("display_screen", false);
            ale[i].setBool("sound", false);
        #endif

        // Load the ROM file. (Also resets the system for new settings to
        // take effect.)
        ale[i].loadROM(argv[1]);
    }

    float totalReward[nthread] = {0};


    // Get the vector of legal actions
    ActionVect legal_actions = ale[0].getLegalActionSet();

    double wall0 = get_wall_time();

    double acts[nthread] = {0};

    // Play 10 episodes
    #pragma omp parallel for schedule(static)
    for (int thr=0; thr<nthread; thr++) {
        for (int episode=0; episode < 10; episode++ ){
            // float totalReward = 0;
            while (!ale[thr].game_over()) {
                Action a = legal_actions[rand() % legal_actions.size()];
                // Apply the action and get the resulting reward
                float reward = ale[thr].act(a);
                totalReward[thr] += reward;
                acts[thr] += 1;
            }
            cout << "Thread " << thr << " Episode " << episode << " ended with score: " << totalReward[thr] << endl;
            ale[thr].reset_game();
            totalReward[thr] = 0.;
        }
    }

    double wall1 = get_wall_time();
    double elapsed = wall1 - wall0;
    double total_acts = 0;
    for(int thr=0; thr<nthread; thr++)
        total_acts += acts[thr];
    double rate = total_acts / elapsed;
    cout << "Wall time: " << wall1 - wall0 << endl;
    cout << "Total actions: " << total_acts << endl;
    cout << "Actions per second: " << rate << endl;

    return 0;
}
