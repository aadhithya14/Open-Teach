/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using UnityEngine;

namespace Oculus.Interaction
{
    public abstract class CandidateComparer<T> : MonoBehaviour, ICandidateComparer where T:class
    {
        public int Compare(object a, object b)
        {
            T typedA = a as T;
            T typedB = b as T;

            if (typedA != null && typedB != null)
            {
                return Compare(typedA, typedB);
            }

            return 0;
        }

        public abstract int Compare(T a, T b);
    }
}
