/*==========================================================================
 * Copyright (c) 2004 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the Lemur Toolkit for Language Modeling and Information Retrieval
 * is subject to the terms of the software license set forth in the LICENSE
 * file included with this software, and also available at
 * http://www.lemurproject.org/license.html
 *
 *==========================================================================
 */


//
// F1LOGTermScoreFunction
//
// F1-LOG score function based on
// Fang, et. al.'s axiomatic approach
// to information retrieval
//
// 30 November 2005 -- dam
//

#ifndef INDRI_F1LOGTERMSCOREFUNCTION_HPP
#define INDRI_F1LOGTERMSCOREFUNCTION_HPP

#include "indri/TermScoreFunction.hpp"
#include <math.h>
#include <iostream>
#ifdef GET_QUERY
#include "indri/Repository.hpp"
#endif
using namespace std;

namespace indri
{
  namespace query
  {
    
    class F1LOGTermScoreFunction : public TermScoreFunction {
    private:
      /// inverse document frequency (IDF) for this term
      double _inverseDocumentFrequency; 
      /// average document length in the collection
      double _averageDocumentLength;

      double _termWeight;

      // These are F1-LOG parameters
      double _s;

      // The following values are precomputed so that score computation will go faster
      //double _tf;
      double _termWeightTimesIDF;

      void _precomputeConstants() {
        //_tf = _s / _averageDocumentLength;
        _termWeightTimesIDF = _termWeight * _inverseDocumentFrequency * ( _averageDocumentLength + _s );
      }

    public:
      F1LOGTermScoreFunction( double idf, double averageDocumentLength, double s = 0.5 ) {
        _inverseDocumentFrequency = idf;
        _averageDocumentLength = averageDocumentLength;

        _s = s;

        _termWeight = 1.0;
        _precomputeConstants();
      }

      double scoreOccurrence( double occurrences, int documentLength ) {
        //
        // Score function is:
        //                              s + avdl
        // score = termWeight * IDF *-------------- * (1+ln(1+ln(occurrence)))
        //                            avdl + s*|D|
        //

        double numerator;
        if( occurrences != 0 ){
            numerator = _termWeightTimesIDF * ( 1 + log( 1+log(occurrences) ) );
        } else{
            numerator = 0;
        }
        double denominator = _averageDocumentLength + _s*documentLength;
        return numerator / denominator;
      }

      #ifdef GET_QUERY
      double scoreOccurrence( double occurrences, int unique_term_counts, int documentLength, int query_length, std::map<std::string, int> uniqueQueryTerms, indri::collection::Repository& r, lemur::api::DOCID_T docID, std::string qTerm ) {
        return scoreOccurrence(occurrences, documentLength);
      }
      #else
      double scoreOccurrence( double occurrences, int unique_term_counts, int documentLength ) {
        return scoreOccurrence(occurrences, documentLength);
      }
      #endif

      double scoreOccurrence( double occurrences, int contextSize, double documentOccurrences, int documentLength ) {
        return scoreOccurrence( occurrences, contextSize );
      }
      
      double maximumScore( int minimumDocumentLength, int maximumOccurrences ) {
        return scoreOccurrence( maximumOccurrences, minimumDocumentLength );
      }
    };
  }
}

#endif // INDRI_F1LOGTERMSCOREFUNCTION_HPP

