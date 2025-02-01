from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from enum import Enum


class CreditRating(Enum):
    """Bond credit rating in ascending risk order (AAA is lowest risk, D is highest risk)"""
    AAA = 1
    AA_PLUS = 2
    AA = 3
    AA_MINUS = 4
    A_PLUS = 5
    A = 6
    A_MINUS = 7
    BBB_PLUS = 8
    BBB = 9
    BBB_MINUS = 10
    BB_PLUS = 11
    BB = 12
    BB_MINUS = 13
    B_PLUS = 14
    B = 15
    B_MINUS = 16
    CCC_PLUS = 17
    CCC = 18
    CCC_MINUS = 19
    CC = 20
    C = 21
    D = 22

    @classmethod
    def from_string(cls, rating: str) -> 'CreditRating':
        """Convert string rating to enum value"""
        # Remove any whitespace and convert to uppercase
        rating = rating.strip().upper()
        
        # Handle modifiers
        rating = rating.replace('+', '_PLUS').replace('-', '_MINUS')
        
        try:
            return cls[rating]
        except KeyError:
            raise ValueError(f"Invalid credit rating: {rating}")

    def display(self) -> str:
        """Return rating in standard format (e.g., 'AA+', 'BBB-')"""
        name = self.name
        return name.replace('_PLUS', '+').replace('_MINUS', '-')

    @staticmethod
    def from_score(score: float) -> 'CreditRating':
        """Convert a rating score back to a CreditRating enum"""
        closest_rating = min(CreditRating, key=lambda x: abs(x.value - score))
        return closest_rating

    def is_investment_grade(self) -> bool:
        """Check if rating is investment grade (BBB- or better)"""
        return self.value <= CreditRating.BBB_MINUS.value

    def __lt__(self, other: 'CreditRating') -> bool:
        """Less than comparison (lower value means lower risk)"""
        if not isinstance(other, CreditRating):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: 'CreditRating') -> bool:
        """Less than or equal comparison"""
        if not isinstance(other, CreditRating):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: 'CreditRating') -> bool:
        """Greater than comparison"""
        if not isinstance(other, CreditRating):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: 'CreditRating') -> bool:
        """Greater than or equal comparison"""
        if not isinstance(other, CreditRating):
            return NotImplemented
        return self.value >= other.value

    @classmethod
    def get_ordered_ratings(cls) -> List['CreditRating']:
        """Get list of ratings in ascending risk order (AAA to D)"""
        return sorted(cls, key=lambda x: x.value)

    @classmethod
    def get_rating_range(cls, start: 'CreditRating', end: 'CreditRating') -> List['CreditRating']:
        """Get list of ratings between start and end (inclusive) in ascending risk order"""
        return [r for r in cls.get_ordered_ratings() if start.value <= r.value <= end.value]

    @property
    def next_rating(self) -> Optional['CreditRating']:
        """Get next worse rating (higher risk). Returns None if already at D."""
        try:
            return CreditRating(self.value + 1)
        except ValueError:
            return None

    @property
    def prev_rating(self) -> Optional['CreditRating']:
        """Get next better rating (lower risk). Returns None if already at AAA."""
        try:
            return CreditRating(self.value - 1)
        except ValueError:
            return None


class RatingGrade(str, Enum):
    INVESTMENT_GRADE = "Investment Grade"
    HIGH_YIELD = "High Yield"

    @staticmethod
    def from_rating(rating: CreditRating) -> 'RatingGrade':
        return RatingGrade.INVESTMENT_GRADE if rating.is_investment_grade() else RatingGrade.HIGH_YIELD


class Bond(BaseModel):
    isin: str
    clean_price: float
    ytm: float
    modified_duration: float
    maturity_date: datetime
    coupon_rate: float
    coupon_frequency: int
    credit_rating: CreditRating
    min_piece: float
    increment_size: float
    currency: str
    day_count_convention: str
    issuer: str
    country: Optional[str] = None
    sector: Optional[str] = None
    payment_rank: Optional[str] = None

    @property
    def rating_grade(self) -> RatingGrade:
        return RatingGrade.from_rating(self.credit_rating)


class PortfolioConstraints(BaseModel):
    total_size: float = Field(..., description="Total portfolio size in base currency")
    min_securities: int = Field(..., description="Minimum number of securities")
    max_securities: int = Field(..., description="Maximum number of securities")
    min_position_size: float = Field(..., description="Minimum position size")
    max_position_size: float = Field(..., description="Maximum position size")
    target_duration: float = Field(..., description="Target portfolio duration")
    duration_tolerance: float = Field(0.5, description="Acceptable deviation from target duration")
    min_rating: CreditRating = Field(..., description="Minimum portfolio rating")
    rating_tolerance: int = Field(1, description="Number of notches tolerance for rating")
    min_yield: float = Field(..., description="Minimum portfolio yield")
    max_issuer_exposure: float = Field(0.1, description="Maximum exposure to single issuer")
    grade_constraints: Dict[RatingGrade, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Grade-specific constraints as (min, max) tuples"
    )
    max_hy_position_size: Optional[float] = Field(None, description="Maximum size for high yield positions")
    sector_constraints: Dict[str, float] = Field(
        default_factory=dict,
        description="Maximum exposure per sector"
    )
    payment_rank_constraints: Dict[str, float] = Field(
        default_factory=dict,
        description="Maximum exposure per payment rank"
    )
    maturity_bucket_constraints: Dict[str, float] = Field(
        default_factory=dict,
        description="Maximum exposure per maturity bucket (format: 'YYYY-YYYY')"
    )


class OptimizationResult(BaseModel):
    success: bool = Field(..., description="Whether the optimization succeeded")
    status: str = Field(..., description="Solver status")
    portfolio: Dict[str, float]
    metrics: Dict[str, float]
    constraints_satisfied: bool
    constraint_violations: List[str]
    optimization_status: str
    solve_time: float
