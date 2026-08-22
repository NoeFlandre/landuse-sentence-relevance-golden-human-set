Feature: Human sentence annotation

  Scenario: Label the current sentence as relevant
    Given the annotation app is running
    When I open the annotation page
    Then I see the sentence and minimal place metadata
    When I choose "Yes — relevant"
    Then the Yes count is 1
