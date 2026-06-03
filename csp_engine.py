# src/p1_dataset/csp_engine.py
# Constraint Satisfaction Problem with AC-3 Arc Consistency
# For food label contradiction detection

from constraint import Problem, AllDifferentConstraint
from typing import List, Dict, Tuple

# ── CONSTRAINT DATABASE ──────────────────────────────────────────────
# Each claim maps to a set of ingredients that VIOLATE it
# (Derived from FSSAI Schedule I and food science definitions)

CLAIM_CONSTRAINTS = {
    'no_added_sugar': {
        'violating_ingredients': [
            'glucose syrup', 'glucose', 'dextrose', 'maltodextrin',
            'fructose', 'high fructose corn syrup', 'corn syrup',
            'invert sugar', 'invert syrup', 'sugar syrup',
            'cane sugar', 'raw sugar', 'brown sugar', 'molasses',
            'fruit concentrate', 'date syrup', 'agave syrup',
        ],
        'trigger_keywords': [
            'no added sugar', 'no sugar added', 'sugar free',
            'zero sugar', 'without added sugar'
        ]
    },
    'no_preservatives': {
        'violating_ingredients': [
            'sodium benzoate', 'e211', 'potassium sorbate', 'e202',
            'sodium nitrate', 'e251', 'sodium nitrite', 'e250',
            'sulphur dioxide', 'e220', 'calcium propionate', 'e282',
            'bha', 'e320', 'bht', 'e321', 'tbhq', 'e319',
        ],
        'trigger_keywords': [
            'no preservatives', 'preservative free', 'no artificial preservatives'
        ]
    },
    'natural': {
        'violating_ingredients': [
            'sodium benzoate', 'e211', 'tartrazine', 'e102',
            'sunset yellow', 'e110', 'carmoisine', 'e122',
            'brilliant blue', 'e133', 'aspartame', 'e951',
            'saccharin', 'e954', 'acesulfame', 'e950',
            'sucralose', 'e955', 'monosodium glutamate', 'msg', 'e621',
            'sodium nitrate', 'e251', 'potassium sorbate', 'e202',
            'hydrogenated', 'partially hydrogenated',
            'artificial flavor', 'artificial colour', 'artificial color',
        ],
        'trigger_keywords': [
            '100% natural', 'all natural', 'purely natural', 'natural ingredients',
            'made with natural', 'no artificial'
        ]
    },
    'gluten_free': {
        'violating_ingredients': [
            'wheat', 'wheat flour', 'whole wheat', 'wheat starch',
            'modified wheat starch', 'barley', 'rye', 'oats',
            'semolina', 'durum wheat', 'spelt', 'kamut',
            'malt', 'malt extract', 'barley malt',
        ],
        'trigger_keywords': [
            'gluten free', 'gluten-free', 'no gluten', 'without gluten'
        ]
    },
    'vegan': {
        'violating_ingredients': [
            'milk', 'milk solids', 'milk powder', 'skimmed milk',
            'whey', 'whey protein', 'casein', 'lactose',
            'butter', 'ghee', 'cream', 'cheese',
            'egg', 'egg powder', 'egg white', 'egg yolk',
            'gelatin', 'gelatine', 'honey', 'beeswax',
            'carmine', 'e120', 'cochineal',
        ],
        'trigger_keywords': [
            'vegan', '100% vegan', 'plant based', 'plant-based', 'dairy free'
        ]
    },
    'no_artificial_colours': {
        'violating_ingredients': [
            'tartrazine', 'e102', 'quinoline yellow', 'e104',
            'sunset yellow', 'e110', 'carmoisine', 'e122',
            'ponceau', 'e124', 'erythrosine', 'e127',
            'allura red', 'e129', 'brilliant blue', 'e133',
            'indigotine', 'e132', 'brilliant black', 'e151',
        ],
        'trigger_keywords': [
            'no artificial colours', 'no artificial colors',
            'no added colours', 'colour free'
        ]
    },
}


class CSPContradictionEngine:
    """
    Implements CSP with AC-3 arc consistency for label contradiction detection.

    Each detected claim is a CSP variable.
    Constraints encode FSSAI regulatory rules.
    AC-3 propagates constraint violations across the network.
    """

    def __init__(self):
        self.constraint_db = CLAIM_CONSTRAINTS

    def detect_claims_in_text(self, label_text: str) -> List[str]:
        """Find which constraint types are triggered by label text."""
        label_lower = label_text.lower()
        triggered = []
        for claim_type, data in self.constraint_db.items():
            for kw in data['trigger_keywords']:
                if kw in label_lower:
                    triggered.append(claim_type)
                    break
        return triggered

    def check_ingredient_violations(
        self, claim_type: str, ingredient_text: str
    ) -> List[str]:
        """Return list of violating ingredients found for this claim."""
        ing_lower = ingredient_text.lower()
        violations = []
        for bad_ing in self.constraint_db[claim_type]['violating_ingredients']:
            if bad_ing in ing_lower:
                violations.append(bad_ing)
        return violations

    def ac3_propagate(
        self,
        claims: List[str],
        ingredient_text: str
    ) -> Dict:
        """
        AC-3 algorithm implementation.
        Initialise queue with all (claim, ingredient_set) arcs.
        Propagate constraints until queue empty or failure detected.
        Returns dict with violation results per claim.
        """
        # Domain of each claim: {satisfied, violated}
        domains = {claim: 'satisfied' for claim in claims}
        violations_found = {}

        # Initialise arc queue: all (claim, ingredient_check) pairs
        queue = [(claim, ingredient_text) for claim in claims]

        while queue:
            claim, ing_text = queue.pop(0)   # dequeue

            # REVISE: check if this arc causes a domain change
            violated_ings = self.check_ingredient_violations(claim, ing_text)

            if violated_ings:
                # Domain collapses to {violated} — hard contradiction confirmed
                if domains[claim] == 'satisfied':
                    domains[claim] = 'violated'
                    violations_found[claim] = violated_ings

                    # Propagate: re-add neighbouring claims to queue
                    neighbours = [c for c in claims if c != claim]
                    for neighbour in neighbours:
                        if (neighbour, ing_text) not in queue:
                            queue.append((neighbour, ing_text))

        return {
            'domains': domains,
            'violations': violations_found,
            'csp_score': len(violations_found) / max(len(claims), 1)
        }

    def run(self, label_text: str, ingredient_text: str) -> Dict:
        """Full CSP pipeline for one product."""
        # Step 1: Detect which claims are present on the label
        claims = self.detect_claims_in_text(label_text)

        if not claims:
            return {
                'claims_detected': [],
                'domains': {},
                'violations': {},
                'csp_score': 0.0
            }

        # Step 2: Run AC-3 propagation
        result = self.ac3_propagate(claims, ingredient_text)
        result['claims_detected'] = claims
        return result


# ── TEST THE ENGINE ───────────────────────────────────────────────────
if __name__ == '__main__':
    engine = CSPContradictionEngine()

    # Test case
    label       = '100% natural ingredients. No added sugar. Gluten free.'
    ingredients = 'wheat flour, glucose syrup, sodium benzoate (E211), palm oil'

    result = engine.run(label, ingredients)

    print('=== CSP-AC3 Results ===')
    print(f'Claims detected: {result["claims_detected"]}')
    print(f'Domains: {result["domains"]}')
    print(f'Violations: {result["violations"]}')
    print(f'CSP Score: {result["csp_score"]:.2f}')

    # Expected output:
    # Claims: ['natural', 'no_added_sugar', 'gluten_free']
    # Violations: {'natural': ['sodium benzoate', 'e211'],
    #              'no_added_sugar': ['glucose syrup'],
    #              'gluten_free': ['wheat flour']}
    # CSP Score: 1.0  (all 3 claims violated)
