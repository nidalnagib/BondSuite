import cvxpy as cp
import logging
import time
from typing import Optional, List, Tuple

# Get logger
logger = logging.getLogger(__name__)

class SolverManager:
    def __init__(self):
        self.primary_solver = 'SCIP'  # SCIP is good for mixed-integer problems
        self.fallback_solvers = ['ECOS_BB', 'GLPK_MI']  # Other mixed-integer capable solvers
        
    def solve(self, problem: cp.Problem, max_iters: int = 10000000) -> Tuple[Optional[str], float]:
        """Attempt to solve the optimization problem with various solvers
        
        Returns:
            Tuple[Optional[str], float]: (solver status, solve time in seconds)
        """
        # Check problem characteristics
        has_integer = any(v.attributes['boolean'] or v.attributes['integer'] 
                         for v in problem.variables())
        has_quadratic = any(expr.is_quadratic() for expr in 
                          [problem.objective.expr] + [c.expr for c in problem.constraints])
        
        logger.info(f"Problem characteristics - Integer vars: {has_integer}, "
                   f"Quadratic: {has_quadratic}")
        
        start_time = time.time()
        
        # Try primary solver first
        try:
            logger.info(f"Attempting solution with {self.primary_solver}")
            result = problem.solve(
                solver=self.primary_solver,
                verbose=True
            )
            solve_time = time.time() - start_time
            if problem.status in ['optimal', 'optimal_inaccurate']:
                logger.info(f"Primary solver {self.primary_solver} succeeded with status: {problem.status}")
                return problem.status, solve_time
            logger.warning(f"Primary solver {self.primary_solver} failed with status: {problem.status}")
        except Exception as e:
            logger.warning(f"Primary solver {self.primary_solver} failed with error: {str(e)}")
        
        # Try fallback solvers
        for solver in self.fallback_solvers:
            try:
                logger.info(f"Attempting solution with fallback solver {solver}")
                solver_opts = {}
                if solver == 'ECOS_BB':
                    solver_opts = {
                        'abstol': 1e-3,
                        'reltol': 1e-3,
                        'feastol': 1e-3,
                        'abstol_inacc': 1e-3,
                        'reltol_inacc': 1e-3,
                        'max_iters': max_iters
                    }
                elif solver == 'GLPK_MI':
                    solver_opts = {
                        'msg_lev': 'GLP_MSG_ON'
                    }
                
                result = problem.solve(
                    solver=solver,
                    verbose=True,
                    **solver_opts
                )
                solve_time = time.time() - start_time
                if problem.status in ['optimal', 'optimal_inaccurate']:
                    logger.info(f"Fallback solver {solver} succeeded with status: {problem.status}")
                    return problem.status, solve_time
                logger.warning(f"Fallback solver {solver} failed with status: {problem.status}")
            except Exception as e:
                logger.warning(f"Fallback solver {solver} failed with error: {str(e)}")
        
        solve_time = time.time() - start_time
        logger.error("All solution attempts failed")
        return 'failed', solve_time
