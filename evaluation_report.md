# IntelliSchedule AI Academic Evaluation Report

Measured on: 2026-11-15 17:27:15 India Standard Time

## 1. Problem definition

The instance schedules 220 weekly meetings from 110 sections across 35 teachers, 16 rooms, and 10 student groups.
Each meeting receives a weekday, contiguous start slot, and compatible room.

## 2. AI algorithms implemented

- **CSP backtracking:** dynamic MRV, degree tie-break, forward-checked domains, bounded nodes/time.
- **Simulated Annealing:** feasible single-meeting neighborhood moves, exponential cooling, Metropolis acceptance, best-so-far tracking.
- **Tabu Search:** sampled feasible neighborhood, reverse-move tabu tenure, aspiration when a move beats the global best.
- **Hill Climbing:** best sampled feasible neighbor; stops after a stagnant sampled-neighborhood limit.
- **Means-Ends repair:** compares current assignments with a desired state and uses bounded CSP repair to minimize disruption.

## 3. Mathematical constraint formulation

For each meeting `i`, choose exactly one placement `x[i]=(day,start,room)`. For every pair of meetings sharing a teacher, student group, or room, occupied slot intervals must be disjoint. Feasible placements also require `room.capacity >= official_enrollment`, room type/equipment compatibility, teacher availability, daily teacher hours within the configured maximum, and a two-period contiguous lab block. Shared elective groups are modeled as a union of occupied student-group resources.

## 4. Soft-constraint objective function

The minimized cost combines weighted student idle gaps, per-group daily-load variance, adjacent heavy/same-subject penalties, teacher preference misses, teacher idle gaps, classroom-load balance, teacher-load variance, and attendance-weighted slot utility. The weights shown in the application are configurable/learned only from explicit preference-feedback rows; attendance prediction does not train preference weights. Predicted attendance never relaxes official enrollment-based room capacity.

## 5. Machine-learning methodology

Attendance and demand Random Forests are trained on the available historical files, or generated synthetic records when none are uploaded. Synthetic-only scores do not demonstrate real-world accuracy. Attendance predictions affect soft slot utility; they do not automatically learn preference weights.
Attendance Random Forest temporal/row-order holdout metrics: MAE 0.0386 attendance fraction, RMSE 0.0494, R² 0.588.
Demand Random Forest holdout metrics: MAE 5.77 students, RMSE 7.61, R² 0.981.

## 6. Experimental setup

- Synthetic demo: 220 weekly meetings; deterministic seed 42.
- CSP: 220 search nodes, 0 backtracks, 24.562s measured CSP wall time.
- All metaheuristics start from the same CSP-feasible seed and use the same objective.
- Each optimizer gets up to 150 iterations or 12 seconds; completed work is reported.
- No optimizer may accept a hard-infeasible neighbor; all final schedules are revalidated.

## 7. Algorithm comparison

| Algorithm | Initial cost | Final cost | Improvement (%) | Soft score | Runtime (s) | Iterations | Student idle | Teacher fairness (%) | Hard violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Traditional CSP seed | 941.232 | 941.232 | 0.00 | 18.95 | 24.562 | 0 | 78 | 81.38 | 0 |
| Simulated Annealing | 941.232 | 884.946 | 5.98 | 19.91 | 4.693 | 150 | 58 | 81.38 | 0 |
| Tabu Search | 941.232 | 805.768 | 14.39 | 21.45 | 4.697 | 150 | 48 | 81.38 | 0 |
| Hill Climbing | 941.232 | 777.822 | 17.36 | 22.05 | 4.542 | 150 | 36 | 81.38 | 0 |

## 8. Results and convergence visualization

On this measured instance, **Hill Climbing** achieved the lowest final cost (777.822) among the zero-violation results. This is an instance-specific observation, not a universal algorithm ranking.

Best-cost trace (lower is better; sampled every 10 points):

- **Simulated Annealing:** 941.2 → 936.8 → 936.8 → 936.8 → 928.7 → 928.7 → 926.3 → 922.8 → 912.2 → 893.4 → 884.9
- **Tabu Search:** 941.2 → 906.0 → 875.8 → 866.7 → 857.7 → 833.8 → 822.0 → 818.3 → 812.7 → 809.2 → 805.8
- **Hill Climbing:** 941.2 → 906.0 → 875.8 → 854.5 → 831.5 → 816.7 → 811.2 → 802.4 → 794.9 → 784.5 → 777.8

The Streamlit Optimization Lab plots complete per-iteration traces and measured improvements interactively. Runtimes vary by hardware and dataset.

## 9. Limitations

- Demo attendance and enrollment data are synthetic; their metrics do not establish real-world prediction quality.
- The weekly discretization uses equal-length slots and omits travel time, multi-campus distances, and individual student enrollment lists.
- CSP and repair search are bounded; failure/timeout means no solution was found within the stated bound, not proof of infeasibility.
- Preference feedback must be explicit, representative, and governed; expert defaults are used when no feedback is supplied.

## 10. Future improvements

Integrate authorized SIS attendance/enrollment histories, add campus travel-time and room accessibility constraints, test on multiple real-world instances, add multi-seed confidence intervals, and benchmark the custom CSP against OR-Tools.
