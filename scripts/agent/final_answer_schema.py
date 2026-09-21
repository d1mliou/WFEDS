"""The structured-output contract for the Axis-3 intervention (pre-registration
`scripts/validation/axis3_structured_output_preregistration.md`).

WHY A SEPARATE MODULE: `agent.py`'s SYSTEM_PROMPT and TOOLS must stay byte-identical
across both experimental arms, so their hashes keep pinning the frozen 2026-09-01
configuration. The contract therefore lives here and rides ONLY on `response_format`,
attached to the post-tool completion call. Nothing here is imported by the free arm.

ONE SOURCE OF TRUTH: this module defines the schema; the published artefact
`scripts/validation/wfeds_final_answer_schema.json` is the same object serialised.
`tests/agent/test_output_contract.py` asserts the two cannot drift.

The rule-7 repair lives in `interpretation.description` (pre-registration 1.5,
Option B): the number prohibition and the 2-4 bullet cap are scoped to the free
interpretation field, and the factual slots carry the numbers. The system prompt is
NOT forked, so SYSTEM_PROMPT_SHA256 stays f005ab5e76d6 in both arms and the pre-tool
call is identical, which is what keeps Stage A and Stage B out of the treatment.
"""

import hashlib
import json

FINAL_ANSWER_SCHEMA = {'type': 'object',
 'additionalProperties': False,
 'required': ['status',
              'initial_state',
              'critical_transitions',
              'final_state',
              'interpretation',
              'limitations'],
 'properties': {'status': {'type': 'string',
                           'enum': ['ok',
                                    'tool_error',
                                    'insufficient_input',
                                    'abstained'],
                           'description': '"ok" = το εργαλείο επέστρεψε αποτέλεσμα και '
                                          'το αφηγείσαι. "tool_error" = το εργαλείο '
                                          'επέστρεψε μήνυμα που ξεκινά με INPUT ERROR '
                                          'ή PIPELINE ERROR. "insufficient_input" = '
                                          'δεν υπάρχουν αρκετά στοιχεία εισόδου για να '
                                          'τρέξει το εργαλείο. "abstained" = δεν '
                                          'απαντάς επί της ουσίας. Όταν το status ΔΕΝ '
                                          'είναι "ok", τα initial_state, '
                                          'critical_transitions και final_state είναι '
                                          'ΚΕΝΟΙ πίνακες και η εξήγηση, μαζί με το '
                                          'αυτούσιο μήνυμα σφάλματος όταν υπάρχει, '
                                          'πηγαίνει στο interpretation.'},
                'initial_state': {'type': 'array',
                                  'description': 'ΑΚΡΙΒΩΣ μία εγγραφή: η ΠΡΩΤΗ γραμμή '
                                                 'του change_hours, δηλαδή η αρχική '
                                                 'κατάσταση κατά την έναρξη. Κενός '
                                                 'πίνακας όταν status != "ok".',
                                  'items': {'type': 'object',
                                            'additionalProperties': False,
                                            'required': ['period',
                                                         'settlements_without_route'],
                                            'properties': {'period': {'type': 'integer',
                                                                      'description': 'Το '
                                                                                     'πεδίο '
                                                                                     '`period` '
                                                                                     'αυτής '
                                                                                     'της '
                                                                                     'γραμμής '
                                                                                     'του '
                                                                                     'change_hours, '
                                                                                     'αυτούσιο.'},
                                                           'settlements_without_route': {'type': 'integer',
                                                                                         'description': 'Το '
                                                                                                        'πεδίο '
                                                                                                        '`cut_off` '
                                                                                                        'αυτής '
                                                                                                        'της '
                                                                                                        'γραμμής '
                                                                                                        'του '
                                                                                                        'change_hours, '
                                                                                                        'αυτούσιο: '
                                                                                                        '"ΟΙΚΙΣΜΟΙ '
                                                                                                        'χωρίς '
                                                                                                        'καμία '
                                                                                                        'διαδρομή '
                                                                                                        'διαφυγής '
                                                                                                        '(ΟΧΙ '
                                                                                                        'δρόμοι)".'}}}},
                'critical_transitions': {'type': 'array',
                                         'description': 'Μία εγγραφή για ΚΑΘΕ γραμμή '
                                                        'του change_hours στην οποία '
                                                        'το cut_off είναι ΜΕΓΑΛΥΤΕΡΟ '
                                                        'από το cut_off της '
                                                        'προηγούμενης γραμμής του '
                                                        'change_hours, με τη σειρά που '
                                                        'εμφανίζονται. Καμία τέτοια '
                                                        'γραμμή δεν επιτρέπεται να '
                                                        'λείπει και καμία άλλη γραμμή '
                                                        'δεν επιτρέπεται να προστεθεί. '
                                                        'Κενός πίνακας όταν status != '
                                                        '"ok".',
                                         'items': {'type': 'object',
                                                   'additionalProperties': False,
                                                   'required': ['period',
                                                                'settlements_without_route'],
                                                   'properties': {'period': {'type': 'integer',
                                                                             'description': 'Το '
                                                                                            'πεδίο '
                                                                                            '`period` '
                                                                                            'αυτής '
                                                                                            'της '
                                                                                            'γραμμής '
                                                                                            'του '
                                                                                            'change_hours, '
                                                                                            'αυτούσιο.'},
                                                                  'settlements_without_route': {'type': 'integer',
                                                                                                'description': 'Το '
                                                                                                               'πεδίο '
                                                                                                               '`cut_off` '
                                                                                                               'αυτής '
                                                                                                               'της '
                                                                                                               'γραμμής '
                                                                                                               'του '
                                                                                                               'change_hours, '
                                                                                                               'αυτούσιο: '
                                                                                                               '"ΟΙΚΙΣΜΟΙ '
                                                                                                               'χωρίς '
                                                                                                               'καμία '
                                                                                                               'διαδρομή '
                                                                                                               'διαφυγής '
                                                                                                               '(ΟΧΙ '
                                                                                                               'δρόμοι)".'}}}},
                'final_state': {'type': 'array',
                                'description': 'ΑΚΡΙΒΩΣ μία εγγραφή, από το '
                                               'αντικείμενο `final_hour` του '
                                               'αποτελέσματος, ΟΧΙ από την τελευταία '
                                               'γραμμή του change_hours, που συχνά '
                                               'αντιστοιχεί σε διαφορετική ώρα. Κενός '
                                               'πίνακας όταν status != "ok".',
                                'items': {'type': 'object',
                                          'additionalProperties': False,
                                          'required': ['period',
                                                       'settlements_without_route',
                                                       'road_segments_removed'],
                                          'properties': {'period': {'type': 'integer',
                                                                    'description': 'Το '
                                                                                   'πεδίο '
                                                                                   '`period` '
                                                                                   'του '
                                                                                   '`final_hour`, '
                                                                                   'αυτούσιο.'},
                                                         'settlements_without_route': {'type': 'integer',
                                                                                       'description': 'Το '
                                                                                                      'πεδίο '
                                                                                                      '`cut_off` '
                                                                                                      'του '
                                                                                                      '`final_hour`, '
                                                                                                      'αυτούσιο: '
                                                                                                      '"ΟΙΚΙΣΜΟΙ '
                                                                                                      'χωρίς '
                                                                                                      'καμία '
                                                                                                      'διαδρομή '
                                                                                                      'διαφυγής '
                                                                                                      '(ΟΧΙ '
                                                                                                      'δρόμοι)".'},
                                                         'road_segments_removed': {'type': 'integer',
                                                                                   'description': 'Το '
                                                                                                  'πεδίο '
                                                                                                  '`edges_removed` '
                                                                                                  'του '
                                                                                                  '`final_hour`, '
                                                                                                  'αυτούσιο: '
                                                                                                  '"ΟΔΙΚΑ '
                                                                                                  'ΤΜΗΜΑΤΑ '
                                                                                                  'που '
                                                                                                  'αφαιρέθηκαν '
                                                                                                  '- '
                                                                                                  'ΜΟΝΟ '
                                                                                                  'τελική '
                                                                                                  'ώρα".'}}}},
                'interpretation': {'type': 'string',
                                   'description': 'Ελεύθερο κείμενο στα ελληνικά: η '
                                                  'ερμηνεία σου. Τι σημαίνει η εξέλιξη '
                                                  'για τις αποφάσεις, σειρά και '
                                                  'χρονισμός εκκένωσης, πού αξίζει '
                                                  'αναχαίτιση. Ισχύουν οι κανόνες του '
                                                  'συστήματος. Ο περιορισμός του '
                                                  'κανόνα 7 (ΜΗΝ επαναλαμβάνεις τα '
                                                  'ίδια νούμερα σε λίστα, όριο 2-4 '
                                                  'σύντομα bullets) αφορά ΜΟΝΟ αυτό το '
                                                  'πεδίο. Τα αριθμητικά πεδία '
                                                  'initial_state, critical_transitions '
                                                  'και final_state τα συμπληρώνεις '
                                                  'πλήρως και αυτούσια από το '
                                                  'αποτέλεσμα του εργαλείου όταν το '
                                                  'status είναι "ok": εκεί ανήκουν τα '
                                                  'νούμερα και ο περιορισμός ΔΕΝ τα '
                                                  'αφορά. Εδώ πηγαίνει και η εξήγηση '
                                                  'όταν το status δεν είναι "ok", '
                                                  'καθώς και το αυτούσιο μήνυμα '
                                                  'σφάλματος του εργαλείου όταν '
                                                  'υπάρχει.'},
                'limitations': {'type': 'string',
                                'description': 'Ελεύθερο κείμενο στα ελληνικά: η '
                                               'υποχρεωτική δήλωση ορίων. Τι δεν '
                                               'αποτυπώνει το μοντέλο και ότι η τελική '
                                               'απόφαση ανήκει στον επιχειρησιακό '
                                               'υπεύθυνο, όχι στο μοντέλο. Δεν '
                                               'επιτρέπεται να είναι κενό.'}}}

# Provenance fingerprint of the contract, the counterpart of SYSTEM_PROMPT_SHA256.
# Any edit to the schema (a description string included, since descriptions are
# instructions) changes this hash, so every structured run is traceable to the
# exact contract that produced it.
FINAL_ANSWER_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(FINAL_ANSWER_SCHEMA, ensure_ascii=False,
               sort_keys=True).encode("utf-8")).hexdigest()

FINAL_ANSWER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "wfeds_final_answer",
        "strict": True,
        "schema": FINAL_ANSWER_SCHEMA,
    },
}


if __name__ == "__main__":
    print(FINAL_ANSWER_SCHEMA_SHA256)
