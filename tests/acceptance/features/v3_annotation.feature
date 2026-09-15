Feature: Resumable V3 annotation UI

  Scenario: Resume fresh V3 labeling without showing seeded V2 rows
    Given the V3 annotation app is running
    When I open the V3 annotation page
    Then I see V3 totals and per-source quotas
    And the seeded V2 sentence is not shown
    When I label the fresh V3 sentence "No — not relevant"
    Then the V3 total progress is "2 / 6"
    And the V3 session contains one fresh label
    When I reload the V3 annotation page
    Then the next fresh V3 sentence is shown

  Scenario: Review, relabel, and remove a fresh V3 annotation
    Given the V3 annotation app is running
    When I open the V3 annotation page
    And I label the fresh V3 sentence "Yes — relevant"
    When I change the V3 saved label to "No"
    Then the V3 Yes target shows one saved label
    When I remove the V3 saved annotation
    Then the V3 fresh review is empty
