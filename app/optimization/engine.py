import cvxpy as cp
import numpy as np
from typing import List, Dict, Optional
import logging
from ..data.models import Bond, PortfolioConstraints, CreditRating, OptimizationResult, RatingGrade
from .solver_manager import SolverManager

# Get logger
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, universe: List[Bond], constraints: PortfolioConstraints):
        self.universe = universe
        self.constraints = constraints
        self.solver_manager = SolverManager()
        logger.info(f"Initializing optimizer with {len(universe)} bonds")
        
    def _setup_variables(self):
        """Setup optimization variables"""
        logger.info("Setting up optimization variables")
        self.weights = cp.Variable(len(self.universe))
        
    def _setup_objective(self):
        """Setup optimization objective"""
        logger.info("Setting up optimization objective")
        
        # Simple yield maximization
        yields = np.array([bond.ytm for bond in self.universe])
        objective = cp.Maximize(yields @ self.weights)
        logger.info(f"Objective created with {len(self.universe)} components")
        
        return objective, []

    def _setup_constraints(self):
        """Setup optimization constraints"""
        constraints = []
        logger.info("Setting up optimization constraints")

        # Portfolio weight constraints
        logger.info("Adding portfolio weight constraints")
        constraints.append(cp.sum(self.weights) == 1)  # Sum of weights = 1
        constraints.append(self.weights >= 0)  # No short selling

        # Duration constraints
        logger.info(f"Adding duration constraints: target={self.constraints.target_duration:.2f}, tolerance={self.constraints.duration_tolerance:.2f}")
        duration_vector = np.array([bond.modified_duration for bond in self.universe])
        min_duration = min(duration_vector)
        max_duration = max(duration_vector)
        logger.info(f"Universe duration range: min={min_duration:.2f}, max={max_duration:.2f}")
        
        target_min = self.constraints.target_duration - self.constraints.duration_tolerance
        target_max = self.constraints.target_duration + self.constraints.duration_tolerance
        if target_min > max_duration or target_max < min_duration:
            logger.warning(f"Duration constraint may be infeasible: target range [{target_min:.2f}, {target_max:.2f}] vs universe range [{min_duration:.2f}, {max_duration:.2f}]")
        
        portfolio_duration = duration_vector @ self.weights
        constraints.append(portfolio_duration <= self.constraints.target_duration + self.constraints.duration_tolerance)
        constraints.append(portfolio_duration >= self.constraints.target_duration - self.constraints.duration_tolerance)

        # Rating constraints
        min_rating_score = float(self.constraints.min_rating.value)  # Convert enum value to float
        max_rating_score = min_rating_score + self.constraints.rating_tolerance
        logger.info(f"Adding rating constraints: min={CreditRating.from_score(min_rating_score).display()}, max={CreditRating.from_score(max_rating_score).display()}")
        
        ratings = [bond.credit_rating.value for bond in self.universe]
        min_rating = min(ratings)
        max_rating = max(ratings)
        logger.info(f"Universe rating range: min={CreditRating.from_score(min_rating).display()}, max={CreditRating.from_score(max_rating).display()}")
        
        if min_rating_score > max_rating:
            logger.warning(f"Rating constraint may be infeasible: minimum required rating {CreditRating.from_score(min_rating_score).display()} is better than best available rating {CreditRating.from_score(max_rating).display()}")
        
        rating_vector = np.array([float(bond.credit_rating.value) for bond in self.universe])  # Convert each rating to float
        portfolio_rating = rating_vector @ self.weights
        constraints.append(portfolio_rating <= max_rating_score)

        # Yield constraint
        logger.info(f"Adding minimum yield constraint: {self.constraints.min_yield:.2%}")
        yield_vector = np.array([bond.ytm for bond in self.universe])
        min_yield = min(yield_vector)
        max_yield = max(yield_vector)
        logger.info(f"Universe yield range: min={min_yield:.2%}, max={max_yield:.2%}")
        
        if self.constraints.min_yield > max_yield:
            logger.warning(f"Yield constraint may be infeasible: minimum required yield {self.constraints.min_yield:.2%} is higher than maximum available yield {max_yield:.2%}")
        
        portfolio_yield = yield_vector @ self.weights
        constraints.append(portfolio_yield >= self.constraints.min_yield)

        # Maximum number of securities constraint
        logger.info(f"Adding security count constraints: min={self.constraints.min_securities}, max={self.constraints.max_securities}")
        if self.constraints.min_securities * self.constraints.min_position_size > 1:
            logger.warning(f"Position size constraint may be infeasible: minimum {self.constraints.min_securities} securities at {self.constraints.min_position_size:.1%} each requires {self.constraints.min_securities * self.constraints.min_position_size:.1%} total")
        if self.constraints.max_securities * self.constraints.max_position_size < 1:
            logger.warning(f"Position size constraint may be infeasible: maximum {self.constraints.max_securities} securities at {self.constraints.max_position_size:.1%} each allows only {self.constraints.max_securities * self.constraints.max_position_size:.1%} total")
        
        binary_vars = cp.Variable(len(self.universe), boolean=True)
        M = 1  # Big M value (1 is sufficient since weights are between 0 and 1)

        # Link binary variables to weights and enforce position size constraints
        logger.info(f"Adding position size constraints: min={self.constraints.min_position_size:.2%}, max={self.constraints.max_position_size:.2%}")
        for i in range(len(self.universe)):
            # Weight must be 0 if binary is 0
            constraints.append(self.weights[i] <= M * binary_vars[i])
            # If binary is 1, weight must be at least min_position_size
            constraints.append(self.weights[i] >= self.constraints.min_position_size * binary_vars[i])
            # Weight cannot exceed max_position_size
            constraints.append(self.weights[i] <= self.constraints.max_position_size)

        # Constraint on number of securities
        constraints.append(cp.sum(binary_vars) <= self.constraints.max_securities)
        constraints.append(cp.sum(binary_vars) >= self.constraints.min_securities)

        # Issuer exposure constraints
        logger.info(f"Adding issuer exposure constraint: max={self.constraints.max_issuer_exposure:.2%}")
        unique_issuers = set(bond.issuer for bond in self.universe)
        logger.info(f"Found {len(unique_issuers)} unique issuers in universe")
        for issuer in unique_issuers:
            issuer_bonds = [bond for bond in self.universe if bond.issuer == issuer]
            if len(issuer_bonds) > 0:
                logger.info(f"Issuer {issuer}: {len(issuer_bonds)} bonds available")
            issuer_indices = [i for i, bond in enumerate(self.universe) if bond.issuer == issuer]
            issuer_exposure = cp.sum(self.weights[issuer_indices])
            constraints.append(issuer_exposure <= self.constraints.max_issuer_exposure)

        # Grade constraints
        if self.constraints.grade_constraints:
            # Only handle High Yield constraints
            if RatingGrade.HIGH_YIELD in self.constraints.grade_constraints:
                min_weight, max_weight = self.constraints.grade_constraints[RatingGrade.HIGH_YIELD]
                logger.info(f"Processing High Yield constraints: min={min_weight:.1%}, max={max_weight:.1%}")
                
                hy_indices = [i for i, bond in enumerate(self.universe) if bond.rating_grade == RatingGrade.HIGH_YIELD]
                logger.info(f"Found {len(hy_indices)} High Yield bonds in universe")
                
                # Only add constraint if we have high yield bonds and constraints are meaningful
                if hy_indices:
                    hy_exposure = cp.sum(self.weights[hy_indices])
                    if min_weight > 0:
                        logger.info(f"Adding minimum High Yield constraint: {min_weight:.1%}")
                        constraints.append(hy_exposure >= min_weight)
                    if max_weight < 1:
                        logger.info(f"Adding maximum High Yield constraint: {max_weight:.1%}")
                        constraints.append(hy_exposure <= max_weight)
                        
                    # Add max position size for HY bonds if specified
                    if self.constraints.max_hy_position_size is not None:
                        logger.info(f"Adding maximum High Yield position size constraint: {self.constraints.max_hy_position_size:.1%}")
                        for i in hy_indices:
                            constraints.append(self.weights[i] <= self.constraints.max_hy_position_size)
                            
                elif min_weight > 0:
                    logger.warning(f"No High Yield bonds available but minimum weight of {min_weight:.1%} required")
                    return []  # Return empty constraints to indicate infeasibility

        # Sector constraints
        if self.constraints.sector_constraints:
            logger.info("Processing sector constraints")
            for sector, max_weight in self.constraints.sector_constraints.items():
                sector_indices = [i for i, bond in enumerate(self.universe) if bond.sector == sector]
                if sector_indices:
                    logger.info(f"Adding {sector} sector constraint: max={max_weight:.1%}")
                    sector_exposure = cp.sum(self.weights[sector_indices])
                    constraints.append(sector_exposure <= max_weight)
                else:
                    logger.warning(f"No bonds found for sector: {sector}")

        # Payment rank constraints
        if self.constraints.payment_rank_constraints:
            logger.info("Processing payment rank constraints")
            for rank, max_weight in self.constraints.payment_rank_constraints.items():
                rank_indices = [i for i, bond in enumerate(self.universe) if bond.payment_rank == rank]
                if rank_indices:
                    logger.info(f"Adding {rank} payment rank constraint: max={max_weight:.1%}")
                    rank_exposure = cp.sum(self.weights[rank_indices])
                    constraints.append(rank_exposure <= max_weight)
                else:
                    logger.warning(f"No bonds found for payment rank: {rank}")

        # Maturity bucket constraints
        if self.constraints.maturity_bucket_constraints:
            logger.info("Processing maturity bucket constraints")
            for bucket, max_weight in self.constraints.maturity_bucket_constraints.items():
                start_year, end_year = map(int, bucket.split('-'))
                bucket_indices = [
                    i for i, bond in enumerate(self.universe)
                    if start_year <= bond.maturity_date.year <= end_year
                ]
                if bucket_indices:
                    logger.info(f"Adding maturity bucket {bucket} constraint: max={max_weight:.1%}")
                    bucket_exposure = cp.sum(self.weights[bucket_indices])
                    constraints.append(bucket_exposure <= max_weight)
                else:
                    logger.warning(f"No bonds found for maturity bucket: {bucket}")

        logger.info(f"Total number of constraints: {len(constraints)}")
        return constraints

    def optimize(self) -> OptimizationResult:
        """Run the optimization"""
        logger.info("Starting optimization")
        
        try:
            # Setup optimization variables
            logger.info("Setting up optimization variables")
            self.weights = cp.Variable(len(self.universe))
            
            # Setup objective and constraints
            objective, additional_constraints = self._setup_objective()
            constraints = self._setup_constraints()
            
            # Check if constraints are empty (indicating infeasibility)
            if not constraints:
                logger.error("Problem is infeasible due to invalid constraints")
                return OptimizationResult(
                    success=False,
                    portfolio={},
                    metrics={},
                    constraints_satisfied=False,
                    constraint_violations=["Invalid constraints - check grade constraints and bond availability"]
                )
            
            constraints.extend(additional_constraints)
            
            # Create and solve problem
            problem = cp.Problem(objective, constraints)
            status, solve_time = self.solver_manager.solve(problem)
            
            if status in ['optimal', 'optimal_inaccurate']:
                # Extract results
                portfolio = {}
                for i, bond in enumerate(self.universe):
                    if self.weights.value[i] > 1e-5:  # Filter out very small positions
                        portfolio[bond.isin] = float(self.weights.value[i])
                
                # Calculate portfolio metrics
                metrics = self._calculate_portfolio_metrics(portfolio)
                
                # Check for constraint violations
                violations = self._check_constraint_violations(portfolio)
                
                return OptimizationResult(
                    success=True,
                    status=status,
                    solve_time=solve_time,
                    portfolio=portfolio,
                    metrics=metrics,
                    constraints_satisfied=len(violations) == 0,
                    constraint_violations=violations,
                    optimization_status=status
                )
            else:
                # Failed optimization
                empty_metrics = {
                    'yield': 0.0,
                    'duration': 0.0,
                    'rating': 0.0,
                    'number_of_securities': 0,
                    'number_of_issuers': 0
                }
                return OptimizationResult(
                    success=False,
                    status=status,
                    solve_time=solve_time,
                    portfolio={},
                    metrics=empty_metrics,
                    constraints_satisfied=False,
                    constraint_violations=["Optimization failed to find a solution"],
                    optimization_status=status
                )
                
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            empty_metrics = {
                'yield': 0.0,
                'duration': 0.0,
                'rating': 0.0,
                'number_of_securities': 0,
                'number_of_issuers': 0
            }
            return OptimizationResult(
                success=False,
                status="error",
                solve_time=0.0,
                portfolio={},
                metrics=empty_metrics,
                constraints_satisfied=False,
                constraint_violations=[f"Error during optimization: {str(e)}"],
                optimization_status="error"
            )

    def _calculate_portfolio_metrics(self, portfolio: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio metrics"""
        weights = np.array([portfolio.get(bond.isin, 0) for bond in self.universe])
        
        # Calculate weighted average yield
        ytms = np.array([bond.ytm for bond in self.universe])
        portfolio_yield = ytms @ weights
        
        # Calculate weighted average duration
        durations = np.array([bond.modified_duration for bond in self.universe])
        portfolio_duration = durations @ weights
        
        # Calculate weighted average rating
        ratings = np.array([bond.credit_rating.value for bond in self.universe])
        portfolio_rating = ratings @ weights
        
        # Calculate number of securities
        num_securities = sum(1 for w in weights if w > 1e-4)

        # Calculate number of issuers
        issuers = set(bond.issuer for bond, w in zip(self.universe, weights) if w > 1e-4)
        num_issuers = len(issuers)
        
        # Calculate grade exposures
        grade_exposures = {}
        for grade in RatingGrade:
            grade_indices = [i for i, bond in enumerate(self.universe) if bond.rating_grade == grade]
            grade_exposures[f"grade_{grade.value}"] = float(sum(weights[i] for i in grade_indices))
        
        return {
            'yield': portfolio_yield,
            'duration': portfolio_duration,
            'rating': portfolio_rating,
            'num_securities': num_securities,
            'num_issuers': num_issuers,
            **grade_exposures
        }

    def _check_constraint_violations(self, portfolio: Dict[str, float]) -> List[str]:
        """Check if portfolio satisfies all constraints"""
        violations = []
        epsilon = 1e-2  # Small tolerance for numerical precision
        
        # Check duration constraints
        portfolio_duration = self._calculate_portfolio_duration(portfolio)
        if portfolio_duration < self.constraints.target_duration - self.constraints.duration_tolerance - epsilon:
            violations.append(
                f"Duration below target range: {portfolio_duration:.2f} < "
                f"{self.constraints.target_duration - self.constraints.duration_tolerance:.2f}"
            )
        elif portfolio_duration > self.constraints.target_duration + self.constraints.duration_tolerance + epsilon:
            violations.append(
                f"Duration above target range: {portfolio_duration:.2f} > "
                f"{self.constraints.target_duration + self.constraints.duration_tolerance:.2f}"
            )
        
        # Check rating constraints
        portfolio_rating = self._calculate_portfolio_rating(portfolio)
        if portfolio_rating > self.constraints.min_rating.value + self.constraints.rating_tolerance:
            violations.append(
                f"Portfolio rating below minimum: {CreditRating.from_score(portfolio_rating).display()} < "
                f"{self.constraints.min_rating.display()}"
            )
        
        # Check number of securities constraints
        num_securities = sum(1 for w in portfolio.values() if w > epsilon)
        if num_securities < self.constraints.min_securities:
            violations.append(f"Too few securities: {num_securities} < {self.constraints.min_securities}")
        elif num_securities > self.constraints.max_securities:
            violations.append(f"Too many securities: {num_securities} > {self.constraints.max_securities}")
        
        # Check position size constraints
        for isin, weight in portfolio.items():
            if weight > self.constraints.max_position_size + epsilon:
                violations.append(f"Position {isin} exceeds maximum size: {weight:.4f} > {self.constraints.max_position_size:.4f}")
            elif weight < self.constraints.min_position_size - epsilon and weight > epsilon:
                violations.append(f"Position {isin} below minimum size: {weight:.4f} < {self.constraints.min_position_size:.4f}")
        
        # Check issuer constraints
        issuer_exposures = {}
        for isin, weight in portfolio.items():
            bond = next(b for b in self.universe if b.isin == isin)
            issuer_exposures[bond.issuer] = issuer_exposures.get(bond.issuer, 0) + weight
        
        for issuer, exposure in issuer_exposures.items():
            if exposure > self.constraints.max_issuer_exposure + epsilon:
                violations.append(
                    f"Issuer {issuer} exposure exceeds maximum: {exposure:.4f} > {self.constraints.max_issuer_exposure:.4f}"
                )
        
        # Check grade constraints
        if self.constraints.grade_constraints:
            # Only handle High Yield constraints
            if RatingGrade.HIGH_YIELD in self.constraints.grade_constraints:
                min_weight, max_weight = self.constraints.grade_constraints[RatingGrade.HIGH_YIELD]
                hy_indices = [i for i, bond in enumerate(self.universe) if bond.rating_grade == RatingGrade.HIGH_YIELD]
                hy_exposure = sum(portfolio.get(bond.isin, 0) for i, bond in enumerate(self.universe) if i in hy_indices)
                
                if min_weight > 0 and hy_exposure < min_weight - epsilon:
                    violations.append(f"Minimum High Yield exposure not met: {hy_exposure:.2%} < {min_weight:.2%}")
                if max_weight < 1 and hy_exposure > max_weight + epsilon:
                    violations.append(f"Maximum High Yield exposure exceeded: {hy_exposure:.2%} > {max_weight:.2%}")
        
        # Check yield constraint
        portfolio_yield = self._calculate_portfolio_yield(portfolio)
        if portfolio_yield < self.constraints.min_yield - epsilon:
            violations.append(f"Minimum yield constraint violated: {portfolio_yield:.4f} < {self.constraints.min_yield:.4f}")
        
        return violations

    def _calculate_portfolio_rating(self, portfolio: Dict[str, float]) -> float:
        """Calculate portfolio rating score"""
        weights = np.array([portfolio.get(bond.isin, 0) for bond in self.universe])
        rating_values = np.array([bond.credit_rating.value for bond in self.universe])
        return float(rating_values @ weights)

    def _calculate_portfolio_duration(self, portfolio: Dict[str, float]) -> float:
        """Calculate portfolio duration"""
        weights = np.array([portfolio.get(bond.isin, 0) for bond in self.universe])
        durations = np.array([bond.modified_duration for bond in self.universe])
        return float(durations @ weights)

    def _calculate_portfolio_yield(self, portfolio: Dict[str, float]) -> float:
        """Calculate portfolio yield"""
        weights = np.array([portfolio.get(bond.isin, 0) for bond in self.universe])
        yields = np.array([bond.ytm for bond in self.universe])
        return float(yields @ weights)
