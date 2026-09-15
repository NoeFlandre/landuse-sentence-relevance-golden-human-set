Feature: V3 annotation seed

  Scenario: Build the exact resumable V3 annotation state
    Given the frozen V2 benchmark and a finalized V3 candidate reservoir
    When I build the V3 annotation seed
    Then the seed has 300 rows and exact source-by-label quotas
    And every V2 cell is reserved and the excluded surplus is explained
    When I rerun the V3 annotation seed
    Then the saved seed is reused without selecting new rows
