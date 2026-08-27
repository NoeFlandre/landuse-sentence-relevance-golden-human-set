Feature: Human sentence annotation

  Scenario: Label the current sentence as relevant
    Given the annotation app is running
    When I open the annotation page
    Then I see the sentence and minimal place metadata
    When I choose "Yes — relevant"
    Then the Yes count is 1

  Scenario: Review, relabel, and remove a saved annotation
    Given the annotation app is running
    When I open the annotation page
    And I choose "Yes — relevant"
    Then the saved annotation list shows 1 record
    When I change the saved label to "No"
    Then the Yes count is 0
    And the No count is 1
    When I remove the saved annotation
    Then the saved annotation list shows 0 records
