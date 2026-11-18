| scenario | clean | noisy-phone |
|---|---|---|
| polite-new-patient | PASS | PASS |
| changes-mind | PASS | PASS |
| vague-caller | PASS | PASS |
| mumbled-emergency | PASS | PASS |
| emergency-mid-booking | PASS | PASS |
| angry-cancel | PASS | **FAIL** |
| asks-for-a-price | PASS | PASS |
| no-symptoms-just-checkup | known | known |
| improvised-busy-parent | PASS | PASS |

15/18 calls passed, 1 failed (0 critical: an emergency not flagged), 2 known trade-off.

- clean: 8/9 passed, 6% of words misheard, 18.5 s per turn
- noisy-phone: 7/9 passed, 24% of words misheard, 17.4 s per turn

Not passed:
- angry-cancel (noisy-phone, FAIL): expected message, got urgent
- no-symptoms-just-checkup (clean, KNOWN): expected booked, got urgent; known: the receptionist errs towards urgent on purpose, so "no swelling" still triggers the safety check
- no-symptoms-just-checkup (noisy-phone, KNOWN): expected booked, got urgent; known: the receptionist errs towards urgent on purpose, so "no swelling" still triggers the safety check
