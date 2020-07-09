## Algorithm to allocate houses to synthetic population

1. Assign `P_COLLECTIVE_X_Y` proportion of population in between ages X and Y to senior residencies
2. Generate `POPULATION_SIZE_REGION / AVG_HOUSEHOLD_SIZE`  number of houses of sizes given by `P_HOUSEHOLD_SIZE`  that follow family types given in `P_FAMILY_TYPE_SIZE_X` for size x.
3. Randomly assign `P_MULTIGENERTIONAL_FAMILY_GIVEN_OTHER_HOUSEHOLDS` proportion of `other` housetypes to be multigenerational
4. Set a priority queue for housetypes with priority as follows - multigenerational, housetypes with kids, and the rest

5. Step 1. Filling houses that needs kids.
  - Sample a housetype
  - Sample a kid of less than `MAX_AGE_CHILDREN` age
  - Sample other kids with `ASSORTATIVITY_STRENGTH` controlling which bins to sample from
  - Sample parents that satisfy `AGE_DIFFERENCE_BETWEEN_PARENT_AND_KID` constraint in regards to their kids
  - Sample grandparents that `AGE_DIFFERENCE_BETWEEN_PARENT_AND_KID` constraint in regards to their kids
  - If there are more houses that needs kids and there are no more kids less than `MAX_AGE_CHILDREN`, expand the search to humans in the age range between - `MAX_AGE_CHILDREN` and `MAX_AGE_WITH_PARENT`

6. Step 2. Filling houses that doesn't need any kids.
  - Sample a housetype
  - Depending on the `housetype.living_arrangement` sample other residents

7. Step 3. Create more houses randomly.
  - If there are more humans that available houses, sample more houses
  - Sample residents for them. 
