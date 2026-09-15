Feature: V3 candidate reservoir preflight

  Scenario: Build a globally unique fresh pool before annotation
    Given a frozen V2 benchmark with reserved cells
    And three streamed V3 sources with enough English candidates
    When I build the V3 candidate pool
    Then every source has at least 2 fresh candidates
    And no candidate uses a V2 H3 cell
    And the V3 quota preflight passes
