Feature: V3 candidate reservoir preflight

  Scenario: Build and safely rerun the oversized candidate reservoir
    Given a V2 seed and three bounded V3 source streams
    When I build the V3 candidate reservoir
    Then every V3 source has two fresh cells in the preflight
    And no selected cell is occupied by V2
    When I rerun the V3 candidate reservoir
    Then the saved reservoir is reused without reading source streams
